import logging
from pathlib import Path
import json
import csv
from typing import List, Dict, Any, Set
import argparse
import concurrent.futures
import os

# Explicitly set TOKENIZERS_PARALLELISM to false to avoid warnings and potential deadlocks/slowdowns in forked processes.
# This addresses the warning: "huggingface/tokenizers: The current process just got forked, after parallelism has already been used..."
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from docling.document_converter import DocumentConverter, PdfFormatOption

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Reduce verbosity of docling's logger
logging.getLogger("docling").setLevel(logging.WARNING)

# # Define the expected CSV field names to ensure consistency when appending
# CSV_FIELD_NAMES = [
#     "id", "text", "source_pdf_filename", "heading", "page_number", 
#     "document_title", "part", "category", "docling_origin_filename",
#     "docling_origin_meta_json"
# ]

def _get_processed_pdfs(progress_file: Path) -> Set[str]:
    """
    Reads the progress file to get a set of already processed PDF paths.
    
    Args:
        progress_file: Path to the progress tracking file
        
    Returns:
        Set of absolute paths of PDFs that have already been processed
    """
    processed_pdfs = set()
    if progress_file.exists():
        try:
            with open(progress_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        processed_pdfs.add(line)
            logger.info(f"Found {len(processed_pdfs)} previously processed PDFs in progress file")
        except Exception as e:
            logger.error(f"Error reading progress file {progress_file}: {e}")
    return processed_pdfs

def _update_processed_pdfs(progress_file: Path, pdf_path: Path) -> None:
    """
    Adds a PDF path to the progress file.
    
    Args:
        progress_file: Path to the progress tracking file
        pdf_path: Path of the PDF that was successfully processed
    """
    try:
        with open(progress_file, 'a', encoding='utf-8') as f:
            f.write(f"{pdf_path.absolute()}\n")
    except Exception as e:
        logger.error(f"Error updating progress file {progress_file}: {e}")

def _write_chunks_to_csv(output_file: Path, chunks: List[Dict[str, Any]], append: bool = False, field_names_to_use: List[str] | None = None) -> List[str] | None:
    """
    Writes or appends chunks to the output CSV file.
    Dynamically determines field names if not provided (for new files) or reads header if appending.
    
    Args:
        output_file: Path to the output CSV file
        chunks: List of chunk dictionaries to write
        append: Whether to append to existing file content. If True and file exists, attempts to match header.
                If False, overwrites or creates new.
        field_names_to_use: Optional. If provided, these field names will be used.
                            Crucial for ensuring consistency across multiple write operations in a single run.
        
    Returns:
        The list of field names used for writing, or None if an error occurred or no chunks.
    """
    if not chunks:
        return field_names_to_use # Return provided field_names or None if nothing to write and no names given

    final_field_names = field_names_to_use
    
    # Determine file mode ('w' or 'a') and effective field names
    mode = 'a' if append and output_file.exists() else 'w' # if append is requested and file exists, use 'a', otherwise 'w'

    if mode == 'a': # Appending to an existing file's content
        if final_field_names:
            # Fields are provided by the caller (e.g. established from a previous write in this run).
            # Verify consistency with the actual file header on disk.
            try:
                with open(output_file, 'r', newline='', encoding='utf-8') as f_read:
                    reader = csv.reader(f_read)
                    existing_header = next(reader)
                    if set(existing_header) != set(final_field_names):
                        logger.error(f"Cannot append to {output_file}. Provided field names {final_field_names} "
                                     f"do not match existing file header {existing_header}.")
                        return None
                    # Use the exact order from the existing header if sets match, for robustness
                    final_field_names = existing_header
            except StopIteration: # File is empty, even if it exists
                logger.info(f"File {output_file} exists but is empty. Will write header as if it's a new file using provided/inferred fields.")
                mode = 'w' # Treat as new file write for header purposes
            except Exception as e:
                logger.error(f"Error reading header from {output_file} for append verification: {e}. "
                             f"Proceeding with provided field names {final_field_names}, but columns might mismatch.")
                # Continue with final_field_names, hoping they are correct.
        else:
            # No field_names_to_use provided by caller, but we are in append mode (e.g. file existed at start of run).
            # Must read header from the file.
            try:
                with open(output_file, 'r', newline='', encoding='utf-8') as f_read:
                    reader = csv.reader(f_read)
                    final_field_names = next(reader)
                logger.info(f"Appending to {output_file}. Using existing header: {final_field_names}")
            except StopIteration: # File is empty
                 logger.info(f"File {output_file} exists but is empty. Will infer fields and write as new file.")
                 mode = 'w' # Switch to write mode to infer and write header
                 # final_field_names will be inferred below for 'w' mode
            except Exception as e:
                logger.error(f"Cannot append. Error reading header from existing file {output_file}: {e}. "
                             "And no explicit field names provided by caller.")
                return None
    
    if mode == 'w': # Writing a new file, or 'a' switched to 'w' because file was empty or didn't exist.
        if not final_field_names: # Infer from chunks if not already set (e.g. from caller or failed append read)
            all_keys = set()
            for chunk in chunks:
                all_keys.update(chunk.keys())
            
            if not all_keys:
                logger.warning(f"No keys found in chunks for {output_file}, cannot infer header.")
                return None

            # Define a preferred order for base keys
            base_keys_ordered = ["id", "text", "source_pdf_filename", "heading", "page_number", 
                                 "document_title", "part", "category", "docling_origin_filename",
                                 "docling_origin_meta_json"]
            
            # Start with base keys that are actually present, in preferred order
            inferred_names = [bk for bk in base_keys_ordered if bk in all_keys]
            
            # Add other keys (not base, not path_segment) sorted alphabetically
            other_keys = sorted([k for k in all_keys if k not in inferred_names and not k.startswith("path_segment_")])
            inferred_names.extend(other_keys)

            # Add path_segment_N keys, sorted numerically by N
            path_segment_keys = sorted([k for k in all_keys if k.startswith("path_segment_")], 
                                       key=lambda x: int(x.split('_')[-1]))
            inferred_names.extend(path_segment_keys)
            
            final_field_names = inferred_names
            logger.info(f"Writing new CSV ({output_file}). Inferred header: {final_field_names}")
        # If final_field_names were already provided (e.g. by caller for a 'w' mode), use them.
        # This path also handles the case where an 'append' to an empty file switched to 'w' with pre-set final_field_names.


    if not final_field_names: # Should be caught by now, but as a safeguard
        logger.error(f"CSV field names for {output_file} could not be determined.")
        return None

    try:
        with open(output_file, mode, newline='', encoding='utf-8') as f:
            # Use extrasaction='raise' to catch if a chunk has fields not in final_field_names.
            # Use restval='' to write an empty string for fields in final_field_names but missing from a chunk.
            writer = csv.DictWriter(f, fieldnames=final_field_names, restval='', extrasaction='raise')
            
            if mode == 'w': # Only write header if we are in 'w' mode (new file, or overwrite)
                writer.writeheader()
            
            for item in chunks:
                # Ensure item conforms to final_field_names (DictWriter handles this with restval and extrasaction)
                writer.writerow(item)
        return final_field_names
    except Exception as e:
        logger.error(f"Error writing {len(chunks)} chunks to {output_file} (mode {mode}): {e}", exc_info=True)
        return None

# Helper function to process a single PDF
# This function will be run in parallel by the ProcessPoolExecutor
def _process_single_pdf(pdf_path: Path, pdf_directory: Path, do_ocr: bool, do_table_structure: bool, chunk_tokenizer_name: str | None, accelerator_device: AcceleratorDevice) -> List[Dict[str, Any]]:
    """
    Processes a single PDF file, chunks it, and returns structured chunk data.
    Initializes its own DocumentConverter and Chunker instances for process safety.
    Accepts do_ocr, do_table_structure, and chunk_tokenizer_name options.
    Dynamically adds path_segment_N fields based on directory structure.
    """
    # Reduced verbosity: Changed from INFO to DEBUG
    logger.debug(f"Worker processing PDF: {pdf_path.name} using {accelerator_device.value}, OCR: {do_ocr}, Table Structure: {do_table_structure}, Tokenizer: {chunk_tokenizer_name or 'default'}")
    
    accelerator_opts = AcceleratorOptions(device=accelerator_device)
    pdf_pipeline_options = PdfPipelineOptions(
        do_ocr=do_ocr, 
        do_table_structure=do_table_structure,
        accelerator_options=accelerator_opts
    )
    doc_converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(
            pipeline_options=pdf_pipeline_options
        )}
    )
    chunker = HybridChunker(tokenizer=chunk_tokenizer_name) if chunk_tokenizer_name else HybridChunker()
    
    single_pdf_chunks: List[Dict[str, Any]] = []

    try:
        docling_document = doc_converter.convert(str(pdf_path)).document
        for i, chunk in enumerate(chunker.chunk(dl_doc=docling_document)):
            chunk_text = chunker.serialize(chunk)
            
            origin_filename = None
            origin_meta_others_json = None
            if chunk.meta and chunk.meta.origin:
                origin_filename = chunk.meta.origin.filename
                origin_dict = dict(chunk.meta.origin)
                if 'filename' in origin_dict:
                    del origin_dict['filename']
                if origin_dict:
                    origin_meta_others_json = json.dumps(origin_dict)

            current_heading = None
            if chunk.meta and chunk.meta.headings and len(chunk.meta.headings) > 0:
                current_heading = chunk.meta.headings[0]

            page_number = None
            if (chunk.meta and 
                chunk.meta.doc_items and 
                len(chunk.meta.doc_items) > 0 and
                chunk.meta.doc_items[0].prov and
                len(chunk.meta.doc_items[0].prov) > 0 and
                hasattr(chunk.meta.doc_items[0].prov[0], 'page_no') and
                chunk.meta.doc_items[0].prov[0].page_no is not None):
                page_number = chunk.meta.doc_items[0].prov[0].page_no
            
            path_derived_metadata: Dict[str, Any] = {}
            try:
                relative_path_to_pdf = pdf_path.relative_to(pdf_directory)
                directory_components = relative_path_to_pdf.parts[:-1] # Exclude filename
                for idx, component_name in enumerate(directory_components):
                    path_derived_metadata[f"path_segment_{idx}"] = component_name
            except ValueError:
                logger.warning(f"Could not determine relative path for {pdf_path} under {pdf_directory}. Path segments missing.")
            except Exception as path_e:
                logger.warning(f"Error extracting path metadata for {pdf_path}: {path_e}")

            part_val = path_derived_metadata.get("path_segment_0")
            category_val = path_derived_metadata.get("path_segment_1")

            chunk_data = {
                "id": f"{pdf_path.stem}-{i}",
                "text": chunk_text,
                "source_pdf_filename": pdf_path.name,
                "heading": current_heading,
                "page_number": page_number,
                "document_title": pdf_path.stem, # Using pdf_path.stem for document_title
                "part": part_val,
                "category": category_val,
                "docling_origin_filename": origin_filename,
                "docling_origin_meta_json": origin_meta_others_json,
            }
            # Add remaining path segments (path_segment_2 onwards, if any)
            remaining_path_segments = {
                k: v for k, v in path_derived_metadata.items()
                if k not in ["path_segment_0", "path_segment_1"]
            }
            chunk_data.update(remaining_path_segments)
            single_pdf_chunks.append(chunk_data)
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path.name} in worker: {e}", exc_info=True)
        # Return empty list for this PDF if an error occurs, to not break the whole batch
        return []
        
    return single_pdf_chunks

def process_pdfs_in_directory(pdf_directory: Path, output_file: Path, ignore_dirs: List[str] = None, use_concurrency: bool = True, do_ocr: bool = False, do_table_structure: bool = False, chunk_tokenizer_name: str | None = None, accelerator_device: AcceleratorDevice = AcceleratorDevice.CPU) -> None:
    """
    Processes all PDF files in the given directory, chunks them,
    and writes the structured chunk data to an output CSV file.
    Can use concurrent workers for speed.
    Accepts an accelerator_device option.
    Supports resuming from previous runs.
    Dynamically determines CSV schema.
    """
    if ignore_dirs is None:
        ignore_dirs = []
    
    progress_file = output_file.with_name(f"{output_file.name}.processed_pdfs.log")
    processed_pdfs = _get_processed_pdfs(progress_file)
    
    pdf_files_all = list(pdf_directory.glob("**/*.pdf"))
    pdf_files = []
    
    for pdf_path in pdf_files_all:
        if any(ignored_dir in pdf_path.parts for ignored_dir in ignore_dirs):
            logger.info(f"Skipping {pdf_path} as it is in an ignored directory.")
            continue
        if str(pdf_path.absolute()) in processed_pdfs:
            logger.info(f"Skipping {pdf_path} as it was already processed.")
            continue
        pdf_files.append(pdf_path)

    if not pdf_files:
        logger.warning(f"No new PDF files to process in {pdf_directory}")
        return

    logger.info(f"Found {len(pdf_files)} new PDF files to process in {pdf_directory}. Concurrency: {use_concurrency}, Accelerator: {accelerator_device.value}, OCR: {do_ocr}, Table Structure: {do_table_structure}, Tokenizer: {chunk_tokenizer_name or 'default'}")

    # Master list of field names for the current CSV file. Established by the first successful write.
    master_csv_fields: List[str] | None = None
    # Tracks if any write has successfully occurred in this run, to determine append_operation status for _write_chunks_to_csv
    has_written_this_run = False
    
    if use_concurrency:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            batch_size = min(10, len(pdf_files)) 
            
            for batch_start in range(0, len(pdf_files), batch_size):
                batch_end = min(batch_start + batch_size, len(pdf_files))
                current_batch = pdf_files[batch_start:batch_end]
                
                logger.info(f"Processing batch of {len(current_batch)} PDFs ({batch_start+1}-{batch_end} of {len(pdf_files)})")
                
                future_to_pdf = {
                    executor.submit(_process_single_pdf, pdf_path, pdf_directory, do_ocr, do_table_structure, chunk_tokenizer_name, accelerator_device): pdf_path 
                    for pdf_path in current_batch
                }
                
                batch_chunks_for_csv = []
                successfully_processed_pdfs_in_batch: List[Path] = []

                for future in concurrent.futures.as_completed(future_to_pdf):
                    pdf_path_completed = future_to_pdf[future]
                    try:
                        chunks_from_pdf = future.result()
                        if chunks_from_pdf:
                            batch_chunks_for_csv.extend(chunks_from_pdf)
                            successfully_processed_pdfs_in_batch.append(pdf_path_completed)
                            logger.info(f"Completed processing for {pdf_path_completed.name}, found {len(chunks_from_pdf)} chunks.")
                        else:
                            logger.info(f"No chunks returned from {pdf_path_completed.name} (it might have failed or produced no chunks).")
                    except Exception as exc:
                        logger.error(f"{pdf_path_completed.name} generated an exception during concurrent execution: {exc}", exc_info=True)
                
                if batch_chunks_for_csv:
                    logger.info(f"Writing {len(batch_chunks_for_csv)} chunks from batch to {output_file}")
                    
                    returned_fields = _write_chunks_to_csv(
                        output_file, 
                        batch_chunks_for_csv, 
                        append=has_written_this_run,
                        field_names_to_use=master_csv_fields
                    )
                    
                    if returned_fields:
                        logger.info(f"Successfully wrote batch chunks to {output_file}")
                        if not has_written_this_run:
                            master_csv_fields = returned_fields # Establish master fields
                            has_written_this_run = True
                        
                        # Update progress for successfully processed PDFs in this batch
                        for pdf_done in successfully_processed_pdfs_in_batch:
                            _update_processed_pdfs(progress_file, pdf_done)
                    else:
                        logger.error(f"Failed to write batch chunks to {output_file}. Progress for this batch not updated.")
    else:  # Sequential processing
        logger.info("Processing PDFs sequentially.")
        accelerator_opts_seq = AcceleratorOptions(device=accelerator_device)
        pdf_pipeline_options_seq = PdfPipelineOptions(
            do_ocr=do_ocr,
            do_table_structure=do_table_structure,
            accelerator_options=accelerator_opts_seq
        )
        doc_converter_seq = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(
                pipeline_options=pdf_pipeline_options_seq
            )}
        )
        chunker_seq = HybridChunker(tokenizer=chunk_tokenizer_name) if chunk_tokenizer_name else HybridChunker()

        for i, pdf_path in enumerate(pdf_files):
            logger.info(f"Processing PDF {i+1}/{len(pdf_files)}: {pdf_path.name} using {accelerator_device.value}, OCR: {do_ocr}, Table Structure: {do_table_structure}, Tokenizer: {chunk_tokenizer_name or 'default'}")
            
            pdf_chunks_for_csv = []
            try:
                docling_document = doc_converter_seq.convert(str(pdf_path)).document
                for j, chunk in enumerate(chunker_seq.chunk(dl_doc=docling_document)):
                    chunk_text = chunker_seq.serialize(chunk)
                    origin_filename = None
                    origin_meta_others_json = None
                    if chunk.meta and chunk.meta.origin:
                        origin_filename = chunk.meta.origin.filename
                        origin_dict = dict(chunk.meta.origin)
                        if 'filename' in origin_dict: del origin_dict['filename']
                        if origin_dict: origin_meta_others_json = json.dumps(origin_dict)

                    current_heading = chunk.meta.headings[0] if chunk.meta and chunk.meta.headings else None
                    page_number = None
                    if (chunk.meta and chunk.meta.doc_items and len(chunk.meta.doc_items) > 0 and
                            chunk.meta.doc_items[0].prov and len(chunk.meta.doc_items[0].prov) > 0 and
                            hasattr(chunk.meta.doc_items[0].prov[0], 'page_no') and
                            chunk.meta.doc_items[0].prov[0].page_no is not None):
                        page_number = chunk.meta.doc_items[0].prov[0].page_no

                    path_derived_metadata: Dict[str, Any] = {}
                    try:
                        relative_path_to_pdf = pdf_path.relative_to(pdf_directory)
                        directory_components = relative_path_to_pdf.parts[:-1]
                        for idx, component_name in enumerate(directory_components):
                            path_derived_metadata[f"path_segment_{idx}"] = component_name
                    except ValueError:
                        logger.warning(f"Could not determine relative path for {pdf_path} under {pdf_directory}")
                    except Exception as path_e:
                        logger.warning(f"Error extracting path metadata for {pdf_path}: {path_e}")
                    
                    part_val = path_derived_metadata.get("path_segment_0")
                    category_val = path_derived_metadata.get("path_segment_1")
                    
                    chunk_data = {
                        "id": f"{pdf_path.stem}-{j}", "text": chunk_text,
                        "source_pdf_filename": pdf_path.name, "heading": current_heading,
                        "page_number": page_number, "document_title": pdf_path.stem,
                        "part": part_val,
                        "category": category_val,
                        "docling_origin_filename": origin_filename,
                        "docling_origin_meta_json": origin_meta_others_json,
                    }
                    # Add remaining path segments (path_segment_2 onwards, if any)
                    remaining_path_segments = {
                        k: v for k, v in path_derived_metadata.items()
                        if k not in ["path_segment_0", "path_segment_1"]
                    }
                    chunk_data.update(remaining_path_segments)
                    pdf_chunks_for_csv.append(chunk_data)
                
                if pdf_chunks_for_csv:
                    logger.info(f"Writing {len(pdf_chunks_for_csv)} chunks from {pdf_path.name} to {output_file}")
                    
                    returned_fields = _write_chunks_to_csv(
                        output_file, 
                        pdf_chunks_for_csv, 
                        append=has_written_this_run,
                        field_names_to_use=master_csv_fields
                    )

                    if returned_fields:
                        logger.info(f"Successfully wrote chunks from {pdf_path.name} to {output_file}")
                        if not has_written_this_run:
                            master_csv_fields = returned_fields
                            has_written_this_run = True
                        _update_processed_pdfs(progress_file, pdf_path)
                    else:
                        logger.error(f"Failed to write chunks from {pdf_path.name}. Progress not updated.")
                else:
                    logger.info(f"No chunks generated from {pdf_path.name}")
            except Exception as e:
                logger.error(f"Error processing PDF {pdf_path.name} sequentially: {e}", exc_info=True)
                continue

    logger.info("Processing complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process PDF files and chunk them.")
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing PDF files to process."
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path to the output CSV file."
    )
    parser.add_argument(
        "--ignore-dirs",
        type=str,
        nargs='+',
        default=[],
        help="List of directory names to ignore (e.g., --ignore-dirs dir1 sub_dir2)."
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="mps",
        choices=["cpu", "mps"],
        help="Accelerator device to use ('cpu' or 'mps'). Default is 'mps'."
    )
    parser.add_argument(
        "--concurrent",
        action="store_true",
        help="Run PDF processing concurrently instead of sequentially."
    )
    parser.add_argument(
        "--do-ocr",
        action="store_true",
        help="Enable OCR during PDF processing. Defaults to False."
    )
    parser.add_argument(
        "--do-table-structure",
        action="store_true",
        help="Enable table structure recognition during PDF processing. Defaults to False."
    )
    parser.add_argument(
        "--chunk-tokenizer",
        type=str,
        default=None,
        help="Specify the tokenizer for HybridChunker (e.g., 'BAAI/bge-small-en-v1.5'). Uses docling default if not specified."
    )
    args = parser.parse_args()

    # Define the input directory for PDFs and the output file path
    pdf_input_dir = Path(args.pdf_input_dir)
    output_csv_file = Path(args.output_file)

    selected_accelerator = AcceleratorDevice.CPU
    if args.accelerator == "mps":
        selected_accelerator = AcceleratorDevice.MPS
        # Attempt to set start method to 'spawn' if MPS is selected, as it's often recommended.
        try:
            import multiprocessing
            if multiprocessing.get_start_method(allow_none=True) != 'spawn':
                multiprocessing.set_start_method('spawn', force=True)
                logger.info("Set multiprocessing start method to 'spawn' for MPS.")
        except Exception as e:
            logger.warning(f"Could not set multiprocessing start method to 'spawn' for MPS: {e}. This might be an issue if not already set, or on certain platforms.")

    if not pdf_input_dir.is_dir():
        logger.error(f"Input PDF directory not found: {pdf_input_dir.resolve()}")
        logger.error("Please ensure the directory exists and contains PDF files.")
    else:
        # Pass the concurrency flag and accelerator from args
        process_pdfs_in_directory(
            pdf_input_dir, 
            output_csv_file, 
            args.ignore_dirs, 
            use_concurrency=args.concurrent,
            do_ocr=args.do_ocr,
            do_table_structure=args.do_table_structure,
            chunk_tokenizer_name=args.chunk_tokenizer,
            accelerator_device=selected_accelerator
        )

    logger.info("Processing complete.")