import logging
from pathlib import Path
import json
import csv
from typing import List, Dict, Any, Set
import argparse
import concurrent.futures
import os
import platform
import warnings
from tqdm.auto import tqdm
from rich.logging import RichHandler
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console
from rich import print as rich_print

# Filter out specific deprecation warnings from docling
warnings.filterwarnings("ignore", category=DeprecationWarning, module="docling")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="docling_core")
# Filter out tokenizer sequence length warnings
warnings.filterwarnings("ignore", message="Token indices sequence length is longer than the specified maximum sequence length")

# Explicitly set TOKENIZERS_PARALLELISM to false to avoid warnings and potential deadlocks/slowdowns in forked processes.
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from docling.chunking import HybridChunker
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from docling.document_converter import DocumentConverter, PdfFormatOption

# Configure rich console
console = Console()

# Configure rich logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger(__name__)

# Reduce verbosity of docling's logger
logging.getLogger("docling").setLevel(logging.WARNING)

warnings.filterwarnings("ignore", category=DeprecationWarning)

def _get_processed_pdfs(progress_file: Path) -> Set[str]:
    """Reads the progress file to get a set of already processed PDF paths."""
    if not progress_file.exists():
        return set()
    try:
        with open(progress_file, 'r', encoding='utf-8') as f:
            processed_pdfs = {line.strip() for line in f if line.strip()}
        logger.info(f"Found [bold cyan]{len(processed_pdfs)}[/] previously processed PDFs")
        return processed_pdfs
    except Exception as e:
        logger.error(f"Error reading progress file: [bold red]{e}[/]")
        return set()

def _update_processed_pdfs(progress_file: Path, pdf_path: Path) -> None:
    """Adds a PDF path to the progress file."""
    try:
        with open(progress_file, 'a', encoding='utf-8') as f:
            f.write(f"{pdf_path.absolute()}\\n")
    except Exception as e:
        logger.error(f"Error updating progress file for [bold]{pdf_path.name}[/]: [bold red]{e}[/]")

def _write_chunks_to_csv(output_file: Path, chunks: List[Dict[str, Any]], append: bool = False, field_names_to_use: List[str] | None = None) -> List[str] | None:
    """Writes or appends chunks to CSV. Dynamically determines/validates header."""
    if not chunks:
        return field_names_to_use

    final_field_names = field_names_to_use
    file_exists_and_not_empty = output_file.exists() and output_file.stat().st_size > 0
    mode = 'a' if append and file_exists_and_not_empty else 'w'

    if mode == 'a':
        try:
            with open(output_file, 'r', newline='', encoding='utf-8') as f_read:
                reader = csv.reader(f_read)
                existing_header = next(reader)
            if final_field_names and set(existing_header) != set(final_field_names):
                logger.error(f"Header mismatch. Provided: [bold]{final_field_names}[/], File: [bold]{existing_header}[/]")
                return None
            final_field_names = existing_header # Use header from file if appending
        except StopIteration: # File exists but is empty
            logger.info(f"File is empty, writing as new")
            mode = 'w' # Fallback to write mode
        except Exception as e:
            logger.error(f"Error reading header for append: [bold red]{e}[/]")
            # If final_field_names is None here, it will be inferred in 'w' mode logic
            if not final_field_names: mode = 'w' # Force infer if no fields and error

    if mode == 'w' and not final_field_names:
        all_keys = {key for chunk in chunks for key in chunk.keys()}
        if not all_keys:
            logger.warning(f"Warning: No keys in chunks, cannot infer header")
            return None
        base_keys = ["id", "text", "source_pdf_filename", "heading", "page_number", "document_title", "docling_origin_filename", "docling_origin_meta_json"]
        final_field_names = [bk for bk in base_keys if bk in all_keys]
        final_field_names.extend(sorted([k for k in all_keys if k not in final_field_names and not k.startswith("path_segment_")]))
        final_field_names.extend(sorted([k for k in all_keys if k.startswith("path_segment_")], key=lambda x: int(x.split('_')[-1])))
        logger.info(f"📊 Determined CSV structure with [bold cyan]{len(final_field_names)}[/] columns. Fields:\n" + "\n".join(f"  [bold]•[/] [bright_blue]{fn}[/bright_blue]" for fn in final_field_names))

    if not final_field_names:
        logger.error("❌ CSV field names could not be determined")
        return None

    try:
        with open(output_file, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=final_field_names, restval='', extrasaction='raise')
            if mode == 'w':
                writer.writeheader()
            writer.writerows(chunks)
        return final_field_names
    except Exception as e:
        logger.error(f"❌ Error writing chunks: [bold red]{e}[/]")
        return None

# Helper function to process a single PDF
# This function will be run in parallel by the ProcessPoolExecutor
def _process_single_pdf(pdf_path: Path, pdf_directory: Path, do_ocr: bool, do_table_structure: bool, chunk_tokenizer_name: str | None, accelerator_device: AcceleratorDevice, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    """Processes a single PDF, chunks it, and returns structured chunk data."""    
    # Filter out specific deprecation warnings from docling within the worker
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="docling")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="docling_core")
    # Filter out tokenizer sequence length warnings within the worker
    warnings.filterwarnings("ignore", message="Token indices sequence length is longer than the specified maximum sequence length")
    
    pdf_pipeline_options = PdfPipelineOptions(
        do_ocr=do_ocr, 
        do_table_structure=do_table_structure,
        accelerator_options=AcceleratorOptions(device=accelerator_device)
    )
    doc_converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)})
    chunker_args = {"chunk_size": chunk_size, "overlap": chunk_overlap}
    if chunk_tokenizer_name:
        chunker_args["tokenizer"] = chunk_tokenizer_name
    chunker = HybridChunker(**chunker_args)
    
    single_pdf_chunks: List[Dict[str, Any]] = []

    try:
        docling_document = doc_converter.convert(str(pdf_path)).document
        for i, chunk in enumerate(chunker.chunk(dl_doc=docling_document)):
            chunk_text = chunker.contextualize(chunk)
            
            origin_filename = None
            origin_meta_others_json = None
            if chunk.meta and chunk.meta.origin:
                origin_filename = chunk.meta.origin.filename
                origin_dict = {k: v for k, v in chunk.meta.origin.model_dump().items() if k != 'filename' and v is not None}
                if origin_dict:
                    origin_meta_others_json = json.dumps(origin_dict)

            current_heading = chunk.meta.headings[0] if chunk.meta and chunk.meta.headings else None
            page_number = None
            if chunk.meta and chunk.meta.doc_items and chunk.meta.doc_items[0].prov and hasattr(chunk.meta.doc_items[0].prov[0], 'page_no'):
                page_number = chunk.meta.doc_items[0].prov[0].page_no
            
            path_derived_metadata: Dict[str, Any] = {}
            try:
                # Path segments are relative to the pdf_directory itself
                relative_path_parts = pdf_path.relative_to(pdf_directory).parts[:-1]
                path_derived_metadata = {f"path_segment_{idx}": name for idx, name in enumerate(relative_path_parts)}
            except ValueError:
                pass
            except Exception:
                pass

            chunk_data = {
                "id": f"{pdf_path.stem}-{i}",
                "text": chunk_text,
                "source_pdf_filename": pdf_path.name,
                "heading": current_heading,
                "page_number": page_number,
                "document_title": pdf_path.stem, 
                "docling_origin_filename": origin_filename,
                "docling_origin_meta_json": origin_meta_others_json,
                **path_derived_metadata # Merge path segments directly
            }
            single_pdf_chunks.append(chunk_data)
            
    except Exception as e:
        logger.error(f"❌ Failed to process [bold]{pdf_path.name}[/]: [bold red]{e}[/]")
        return [] # Return empty on error for this PDF
        
    return single_pdf_chunks

def process_pdfs_in_directory(pdf_directory: Path, output_file: Path, ignore_dirs: List[str] = None, use_concurrency: bool = True, do_ocr: bool = False, do_table_structure: bool = False, chunk_tokenizer_name: str | None = None, accelerator_device: AcceleratorDevice = AcceleratorDevice.CPU, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
    """Processes PDFs in a directory, chunks them, and writes to CSV, with concurrency and resume support."""
    ignore_dirs = ignore_dirs or []
    progress_file = output_file.with_name(f"{output_file.name}.processed_pdfs.log")
    processed_pdf_paths = _get_processed_pdfs(progress_file)
    
    all_pdf_files = list(pdf_directory.glob("**/*.pdf"))
    pdf_files_to_process = [
        pdf_path for pdf_path in all_pdf_files
        if not any(ignored_dir in pdf_path.parts for ignored_dir in ignore_dirs) and 
           str(pdf_path.absolute()) not in processed_pdf_paths
    ]

    # Display skipped files info
    skipped_count = 0
    for pdf_path in all_pdf_files:
        if str(pdf_path.absolute()) in processed_pdf_paths:
            skipped_count += 1
        elif any(ignored_dir in pdf_path.parts for ignored_dir in ignore_dirs):
            skipped_count += 1

    if skipped_count > 0:
        logger.info(f"Skipping [bold yellow]{skipped_count}[/] PDFs (already processed or in ignored directories)")

    if not pdf_files_to_process:
        logger.warning("Warning: [bold yellow]No new PDF files to process[/]")
        return

    # Create a summary of the processing configuration
    config_summary = [
        f"Directory: [bold cyan]{pdf_directory}[/]",
        f"Concurrency: [bold]{'Enabled' if use_concurrency else 'Disabled'}[/]",
        f"Accelerator: [bold cyan]{accelerator_device.value}[/]",
        f"OCR: [bold]{'Enabled' if do_ocr else 'Disabled'}[/]",
        f"Table Structure: [bold]{'Enabled' if do_table_structure else 'Disabled'}[/]",
        f"Tokenizer: [bold cyan]{chunk_tokenizer_name or 'default'}[/]",
        f"Chunk Size: [bold cyan]{chunk_size}[/]",
        f"Overlap: [bold cyan]{chunk_overlap}[/]"
    ]
    
    logger.info(f"[bold green]Processing {len(pdf_files_to_process)} PDFs[/]")
    for config_line in config_summary:
        logger.info(config_line)

    master_csv_fields: List[str] | None = None
    has_written_this_run = False
    
    if use_concurrency:
        cpu_cores = os.cpu_count() or 1
        max_workers = min(4, cpu_cores // 2 if cpu_cores > 1 else 1)
        logger.info(f"Using [bold cyan]{max_workers}[/] worker processes")

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_pdf = {
                executor.submit(_process_single_pdf, pdf_path, pdf_directory, do_ocr, do_table_structure, chunk_tokenizer_name, accelerator_device, chunk_size, chunk_overlap): pdf_path 
                for pdf_path in pdf_files_to_process
            }
            
            all_chunks_for_csv = []
            successfully_processed_pdfs_in_run: List[Path] = []

            # Create progress bar for concurrent processing
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Processing PDFs...", total=len(pdf_files_to_process))
                
                for i, future in enumerate(concurrent.futures.as_completed(future_to_pdf)):
                    pdf_path_completed = future_to_pdf[future]
                    try:
                        chunks_from_pdf = future.result()
                        if chunks_from_pdf:
                            all_chunks_for_csv.extend(chunks_from_pdf)
                            successfully_processed_pdfs_in_run.append(pdf_path_completed)
                            progress.update(task, advance=1, description=f"[cyan]Processing PDFs... ({i+1}/{len(pdf_files_to_process)})")
                        else:
                            progress.update(task, advance=1, description=f"[yellow]Processing PDFs... ({i+1}/{len(pdf_files_to_process)})")
                    except Exception as exc:
                        logger.error(f"❌ PDF [bold]{pdf_path_completed.name}[/] failed: [bold red]{exc}[/]")
                        progress.update(task, advance=1, description=f"[red]Processing PDFs... ({i+1}/{len(pdf_files_to_process)})")
            
            if all_chunks_for_csv:
                logger.info(f"Writing [bold cyan]{len(all_chunks_for_csv)}[/] chunks from [bold cyan]{len(successfully_processed_pdfs_in_run)}[/] PDFs")
                returned_fields = _write_chunks_to_csv(
                    output_file, 
                    all_chunks_for_csv, 
                    append=output_file.exists() and output_file.stat().st_size > 0,
                    field_names_to_use=master_csv_fields
                )
                
                if returned_fields:
                    logger.info(f"[bold green]Successfully wrote data to {output_file.name}[/]")
                    if not has_written_this_run:
                        master_csv_fields = returned_fields 
                        has_written_this_run = True
                    
                    for pdf_done in successfully_processed_pdfs_in_run:
                        _update_processed_pdfs(progress_file, pdf_done)
                else:
                    logger.error(f"[bold red]Failed to write data to {output_file.name}[/]")

    else:  # Sequential processing
        logger.info("Processing PDFs sequentially")
        
        # Create progress bar for sequential processing
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]Processing PDFs...", total=len(pdf_files_to_process))
            
            for i, pdf_path in enumerate(pdf_files_to_process):
                progress.update(task, description=f"[cyan]Processing: {pdf_path.name}")
                
                # Call _process_single_pdf directly for sequential processing
                pdf_chunks = _process_single_pdf(pdf_path, pdf_directory, do_ocr, do_table_structure, chunk_tokenizer_name, accelerator_device, chunk_size, chunk_overlap)
                
                if pdf_chunks:
                    should_append_this_pdf = has_written_this_run or (output_file.exists() and output_file.stat().st_size > 0)
                    
                    returned_fields = _write_chunks_to_csv(
                        output_file, 
                        pdf_chunks, 
                        append=should_append_this_pdf,
                        field_names_to_use=master_csv_fields
                    )

                    if returned_fields:
                        if not has_written_this_run:
                            master_csv_fields = returned_fields
                            has_written_this_run = True
                        _update_processed_pdfs(progress_file, pdf_path)
                        progress.update(task, advance=1, description=f"[green]Processed: {pdf_path.name}")
                    else:
                        logger.error(f"❌ Failed to write chunks from [bold]{pdf_path.name}[/]")
                        progress.update(task, advance=1, description=f"[red]Failed: {pdf_path.name}")
                else:
                    progress.update(task, advance=1, description=f"[yellow]No chunks: {pdf_path.name}")

    logger.info("[bold green]Processing complete![/]")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process PDF files, chunk them, and output to CSV.")
    parser.add_argument("input_dir", type=str, help="Directory containing PDF files.")
    parser.add_argument("output_file", type=str, help="Path to the output CSV file.")
    parser.add_argument("--ignore-dirs", type=str, nargs='+', default=[], help="Directory names to ignore.")
    parser.add_argument("--accelerator", type=str, default="cpu", choices=["cpu", "mps"], help="Accelerator: 'cpu' or 'mps'. Default: 'cpu'.")
    parser.add_argument("--concurrent", action="store_true", help="Enable concurrent PDF processing.")
    parser.add_argument("--do-ocr", action="store_true", help="Enable OCR during PDF processing.")
    parser.add_argument("--do-table-structure", action="store_true", help="Enable table structure recognition.")
    parser.add_argument("--chunk-tokenizer", type=str, default=None, help="Tokenizer for HybridChunker (e.g., 'BAAI/bge-small-en-v1.5').")
    parser.add_argument("--chunk-size", type=int, default=500, help="Target chunk size. Default: 500.")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Chunk overlap. Default: 100.")

    args = parser.parse_args()

    # Use rich to print a nice welcome message
    rich_print("\n[bold cyan]DocuNomic PDF Processor[/]\n")

    # Set multiprocessing start method to 'spawn' if not already set, especially for macOS/MPS.
    try:
        import multiprocessing
        current_method = multiprocessing.get_start_method(allow_none=True)
        if current_method != 'spawn':
            force_spawn = True if current_method else False
            multiprocessing.set_start_method('spawn', force=force_spawn)
            logger.info(f"Multiprocessing start method: [bold cyan]spawn[/] (was: {current_method})")
        else:
            logger.info(f"Multiprocessing start method: [bold cyan]spawn[/]")
    except Exception as e:
        logger.warning(f"Could not set multiprocessing start method: [bold yellow]{e}[/]")

    selected_accelerator = AcceleratorDevice.MPS if args.accelerator == "mps" else AcceleratorDevice.CPU

    pdf_input_dir = Path(args.input_dir)
    output_csv_file = Path(args.output_file)

    if not pdf_input_dir.is_dir():
        logger.error(f"❌ [bold red]Input directory not found: {pdf_input_dir.resolve()}[/]")
    else:
        process_pdfs_in_directory(
            pdf_input_dir, 
            output_csv_file, 
            args.ignore_dirs, 
            use_concurrency=args.concurrent,
            do_ocr=args.do_ocr,
            do_table_structure=args.do_table_structure,
            chunk_tokenizer_name=args.chunk_tokenizer,
            accelerator_device=selected_accelerator,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )