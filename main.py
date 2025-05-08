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

# Define the expected CSV field names to ensure consistency when appending
CSV_FIELD_NAMES = [
    "id", "text", "source_pdf_filename", "heading", "page_number", 
    "document_title", "part", "category", "docling_origin_filename",
    "docling_origin_meta_json"
]

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

def _write_chunks_to_csv(output_file: Path, chunks: List[Dict[str, Any]], append: bool = False) -> bool:
    """
    Writes or appends chunks to the output CSV file.
    
    Args:
        output_file: Path to the output CSV file
        chunks: List of chunk dictionaries to write
        append: Whether to append to existing file (True) or create new file (False)
        
    Returns:
        True if successful, False if there was an error
    """
    if not chunks:
        return True  # Nothing to write, consider it successful
        
    mode = 'a' if append and output_file.exists() else 'w'
    try:
        with open(output_file, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELD_NAMES)
            
            # Only write header if we're creating a new file
            if mode == 'w':
                writer.writeheader()
                
            for item in chunks:
                writer.writerow(item)
        return True
    except Exception as e:
        logger.error(f"Error writing chunks to {output_file}: {e}", exc_info=True)
        return False

# Helper function to process a single PDF
# This function will be run in parallel by the ProcessPoolExecutor
def _process_single_pdf(pdf_path: Path, pdf_directory: Path, do_ocr: bool, do_table_structure: bool, chunk_tokenizer_name: str | None, accelerator_device: AcceleratorDevice) -> List[Dict[str, Any]]:
    """
    Processes a single PDF file, chunks it, and returns structured chunk data.
    Initializes its own DocumentConverter and Chunker instances for process safety.
    Accepts do_ocr, do_table_structure, and chunk_tokenizer_name options.
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
            
            part_info = None
            category_info = None
            document_title = pdf_path.stem

            try:
                relative_path = pdf_path.relative_to(pdf_directory)
                relative_path_parts = relative_path.parts
                
                if len(relative_path_parts) > 1:
                    if relative_path_parts[0].startswith("Part_"):
                        part_info = relative_path_parts[0]
                    
                    if len(relative_path_parts) > 2:
                         category_info = relative_path_parts[1]
            except ValueError:
                logger.warning(f"Could not determine relative path for {pdf_path} relative to {pdf_directory} in worker.")
            except Exception as path_e:
                logger.warning(f"Error extracting path metadata for {pdf_path} in worker: {path_e}")

            chunk_data = {
                "id": f"{pdf_path.stem}-{i}",
                "text": chunk_text,
                "source_pdf_filename": pdf_path.name,
                "heading": current_heading,
                "page_number": page_number,
                "document_title": document_title,
                "part": part_info,
                "category": category_info,
                "docling_origin_filename": origin_filename,
                "docling_origin_meta_json": origin_meta_others_json,
            }
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
    Now accepts do_ocr, do_table_structure, and chunk_tokenizer_name.
    """
    if ignore_dirs is None:
        ignore_dirs = []
    
    # Define the progress tracking file path
    progress_file = output_file.with_name(f"{output_file.name}.processed_pdfs.log")
    
    # Load the set of already processed PDF paths
    processed_pdfs = _get_processed_pdfs(progress_file)
    
    # These options are now passed as arguments
    # do_ocr_option = False (replaced by do_ocr argument)
    # do_table_structure_option = False (replaced by do_table_structure argument)
    
    # Find all PDF files
    pdf_files_all = list(pdf_directory.glob("**/*.pdf"))
    pdf_files = []
    
    # Filter PDFs to exclude ones in ignored directories and already processed ones
    for pdf_path in pdf_files_all:
        # Skip if in ignored directory
        if any(ignored_dir in pdf_path.parts for ignored_dir in ignore_dirs):
            logger.info(f"Skipping {pdf_path} as it is in an ignored directory.")
            continue
            
        # Skip if already processed
        if str(pdf_path.absolute()) in processed_pdfs:
            logger.info(f"Skipping {pdf_path} as it was already processed.")
            continue
            
        pdf_files.append(pdf_path)

    if not pdf_files:
        logger.warning(f"No new PDF files to process in {pdf_directory}")
        return

    logger.info(f"Found {len(pdf_files)} new PDF files to process in {pdf_directory}. Concurrency: {use_concurrency}, Accelerator: {accelerator_device.value}, OCR: {do_ocr}, Table Structure: {do_table_structure}, Tokenizer: {chunk_tokenizer_name or 'default'}")

    # Check if output file exists to determine if we should append
    should_append = output_file.exists()
    
    if use_concurrency:
        # Using ProcessPoolExecutor for CPU-bound tasks (PDF parsing can be CPU intensive)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Process PDFs in batches to allow incremental saving
            batch_size = min(10, len(pdf_files))  # Process up to 10 PDFs at a time
            
            for batch_start in range(0, len(pdf_files), batch_size):
                batch_end = min(batch_start + batch_size, len(pdf_files))
                current_batch = pdf_files[batch_start:batch_end]
                
                logger.info(f"Processing batch of {len(current_batch)} PDFs ({batch_start+1}-{batch_end} of {len(pdf_files)})")
                
                # Submit batch of PDF processing tasks
                future_to_pdf = {
                    executor.submit(_process_single_pdf, pdf_path, pdf_directory, do_ocr, do_table_structure, chunk_tokenizer_name, accelerator_device): pdf_path 
                    for pdf_path in current_batch
                }
                
                batch_chunks = []
                for future in concurrent.futures.as_completed(future_to_pdf):
                    pdf_path_completed = future_to_pdf[future]
                    try:
                        chunks_from_pdf = future.result()
                        if chunks_from_pdf:  # Only process if chunks were successfully produced
                            batch_chunks.extend(chunks_from_pdf)
                            logger.info(f"Completed processing for {pdf_path_completed.name}, found {len(chunks_from_pdf)} chunks.")
                            
                            # Update progress immediately for this PDF
                            _update_processed_pdfs(progress_file, pdf_path_completed)
                        else:
                            logger.info(f"No chunks returned from {pdf_path_completed.name} (it might have failed or produced no chunks).")
                    except Exception as exc:
                        logger.error(f"{pdf_path_completed.name} generated an exception during concurrent execution: {exc}", exc_info=True)
                
                # Write this batch's chunks to the CSV
                if batch_chunks:
                    logger.info(f"Writing {len(batch_chunks)} chunks from batch to {output_file}")
                    if _write_chunks_to_csv(output_file, batch_chunks, append=should_append):
                        logger.info(f"Successfully wrote batch chunks to {output_file}")
                        should_append = True  # Future writes should append
    else:  # Sequential processing
        logger.info("Processing PDFs sequentially.")
        # Initialize DocumentConverter and Chunker once for sequential mode.
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
            
            pdf_chunks = []
            try:
                docling_document = doc_converter_seq.convert(str(pdf_path)).document
                for j, chunk in enumerate(chunker_seq.chunk(dl_doc=docling_document)):
                    chunk_text = chunker_seq.serialize(chunk)
                    origin_filename = None
                    origin_meta_others_json = None
                    if chunk.meta and chunk.meta.origin:
                        origin_filename = chunk.meta.origin.filename
                        origin_dict = dict(chunk.meta.origin)
                        if 'filename' in origin_dict:
                            del origin_dict['filename']
                        if origin_dict:
                            origin_meta_others_json = json.dumps(origin_dict)

                    current_heading = chunk.meta.headings[0] if chunk.meta and chunk.meta.headings else None
                    page_number = None
                    if (chunk.meta and chunk.meta.doc_items and len(chunk.meta.doc_items) > 0 and
                            chunk.meta.doc_items[0].prov and len(chunk.meta.doc_items[0].prov) > 0 and
                            hasattr(chunk.meta.doc_items[0].prov[0], 'page_no') and
                            chunk.meta.doc_items[0].prov[0].page_no is not None):
                        page_number = chunk.meta.doc_items[0].prov[0].page_no

                    part_info, category_info = None, None
                    document_title = pdf_path.stem
                    try:
                        relative_path = pdf_path.relative_to(pdf_directory)
                        relative_path_parts = relative_path.parts
                        if len(relative_path_parts) > 1 and relative_path_parts[0].startswith("Part_"):
                            part_info = relative_path_parts[0]
                        if len(relative_path_parts) > 2:
                            category_info = relative_path_parts[1]
                    except ValueError:
                        logger.warning(f"Could not determine relative path for {pdf_path} relative to {pdf_directory}")
                    except Exception as path_e:
                        logger.warning(f"Error extracting path metadata for {pdf_path}: {path_e}")

                    chunk_data = {
                        "id": f"{pdf_path.stem}-{j}", "text": chunk_text,
                        "source_pdf_filename": pdf_path.name, "heading": current_heading,
                        "page_number": page_number, "document_title": document_title,
                        "part": part_info, "category": category_info,
                        "docling_origin_filename": origin_filename,
                        "docling_origin_meta_json": origin_meta_others_json,
                    }
                    pdf_chunks.append(chunk_data)
                
                # Write chunks from this PDF immediately
                if pdf_chunks:
                    logger.info(f"Writing {len(pdf_chunks)} chunks from {pdf_path.name} to {output_file}")
                    if _write_chunks_to_csv(output_file, pdf_chunks, append=should_append):
                        logger.info(f"Successfully wrote chunks from {pdf_path.name} to {output_file}")
                        should_append = True  # Future writes should append
                        
                        # Update progress for this PDF
                        _update_processed_pdfs(progress_file, pdf_path)
                else:
                    logger.info(f"No chunks generated from {pdf_path.name}")
            except Exception as e:
                logger.error(f"Error processing PDF {pdf_path.name} sequentially: {e}", exc_info=True)
                continue  # Skip to the next file

    logger.info("Processing complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process PDF files and chunk them.")
    parser.add_argument(
        "pdf_input_dir",
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