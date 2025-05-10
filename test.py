import unittest
import tempfile
import shutil
# import json # No longer needed directly in this test after changes
import csv
from pathlib import Path
import time # Added for timing
import os # For listing files if needed, though Path.glob should suffice
import multiprocessing # Added for setting start method
import torch # Added for MPS check
import logging # Added for logging
import warnings # Added for warnings
from rich.logging import RichHandler
from rich.console import Console
from rich import print as rich_print
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn

# Filter out tokenizer sequence length warnings
warnings.filterwarnings("ignore", message="Token indices sequence length is longer than the specified maximum sequence length")

# Filter out all DeprecationWarnings globally
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Assuming run.py is in the same directory or accessible in PYTHONPATH
from main import process_pdfs_in_directory 
from docling.datamodel.pipeline_options import AcceleratorDevice # Added import

# Configure rich console
console = Console()

# Configure the logger with rich formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)
logger = logging.getLogger(__name__)

class TestPdfProcessing(unittest.TestCase):

    def setUp(self):
        """Set up temporary directories and copy test PDFs."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.input_pdf_dir = self.test_dir / "input_pdfs"
        self.input_pdf_dir.mkdir()

        # Structured input for path segment testing
        self.structured_input_dir = self.input_pdf_dir / "Part_I" / "Affiliate_Agreements"
        self.structured_input_dir.mkdir(parents=True, exist_ok=True)
        
        self.cpu_sequential_output_file = self.test_dir / "output_chunks_cpu_sequential.csv"
        self.cpu_concurrent_output_file = self.test_dir / "output_chunks_cpu_concurrent.csv"
        self.mps_sequential_output_file = self.test_dir / "output_chunks_mps_sequential.csv"
        self.mps_concurrent_output_file = self.test_dir / "output_chunks_mps_concurrent.csv"

        self.source_agreements_dir = Path("data/CUAD_v1/full_contract_pdf/Part_I/Affiliate_Agreements/")
        if not self.source_agreements_dir.is_dir():
            raise FileNotFoundError(f"Source PDF directory not found: {self.source_agreements_dir}")

        self.source_pdf_names = set()
        for source_pdf_path in self.source_agreements_dir.glob("*.pdf"):
            if source_pdf_path.is_file(): # Ensure it's a file, not a dir ending in .pdf
                target_pdf_path = self.structured_input_dir / source_pdf_path.name
                shutil.copy(source_pdf_path, target_pdf_path)
                self.source_pdf_names.add(source_pdf_path.name)
        
        if not self.source_pdf_names:
            raise FileNotFoundError(f"No PDF files copied from source directory: {self.source_agreements_dir}")
        
        rich_print(f"[bold green]Set up test with [bold cyan]{len(self.source_pdf_names)}[/] PDF files from [cyan]{self.input_pdf_dir}[/]")

    def tearDown(self):
        """Remove temporary directories after the test."""
        shutil.rmtree(self.test_dir)

    def _validate_csv_output(self, output_file: Path, expected_pdf_names: set):
        """Helper function to validate CSV output structure and content."""
        self.assertTrue(output_file.exists() and output_file.is_file(), f"Output CSV {output_file.name} not created or not a file.")

        chunks_data = []
        try:
            with open(output_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                # Validate headers once based on the DictReader's fieldnames
                expected_base_headers = ["id", "text", "source_pdf_filename", "heading", "page_number", 
                                     "document_title", "docling_origin_filename", "docling_origin_meta_json"]
                # Path segments path_segment_0, path_segment_1 are expected from the structured input
                # Their presence is implicitly checked if they are in reader.fieldnames and then per chunk
                # For a more dynamic check, we could infer them or pass them based on self.structured_input_dir
                
                # Check that all expected base headers are present in the CSV file itself.
                # The `_write_chunks_to_csv` should handle path_segment_N dynamically.
                # For this test, we know specific path_segments are expected due to the input structure.
                # So we check base + specific path segments based on test setup knowledge.
                expected_fieldnames = set(expected_base_headers + ["path_segment_0", "path_segment_1"])
                self.assertTrue(expected_fieldnames.issubset(set(reader.fieldnames if reader.fieldnames else [])), 
                                f"CSV {output_file.name} is missing expected headers. Expected at least: {expected_fieldnames}. Got: {reader.fieldnames}")

                for row in reader:
                    # Convert page_number: if it's a digit string, convert to int; if empty or None, set to None.
                    pn = row.get("page_number")
                    row["page_number"] = int(pn) if pn and pn.isdigit() else None
                    chunks_data.append(row)
        except FileNotFoundError:
            self.fail(f"Output CSV file {output_file.name} not found during read.") # Should be caught by initial assert
        except Exception as e:
            self.fail(f"Error reading/parsing CSV {output_file.name}: {e}")
        
        self.assertTrue(chunks_data, f"No chunks in {output_file.name}.")

        processed_pdf_filenames = {chunk["source_pdf_filename"] for chunk in chunks_data}
        self.assertEqual(expected_pdf_names, processed_pdf_filenames,
                         f"Processed PDF filenames in {output_file.name} mismatch. Expected: {expected_pdf_names}, Got: {processed_pdf_filenames}")

        for chunk in chunks_data: 
            # All expected headers (including dynamic path_segment_N) should be keys in each chunk dictionary
            # This was partially checked with reader.fieldnames, but DictReader ensures keys exist if fieldnames were used.
            # However, we need to ensure our *specific expected* fields are there.
            for header in expected_fieldnames: # Check against the combined set
                self.assertIn(header, chunk, f"CSV data in {output_file.name} missing field '{header}' in chunk ID {chunk.get('id', 'N/A')}.")
            
            if chunk.get("page_number") is not None:
                 self.assertIsInstance(chunk["page_number"], int, f"'page_number' in {output_file.name} (chunk {chunk.get('id')}) should be int or None.")

            pdf_stem = Path(chunk["source_pdf_filename"]).stem
            self.assertTrue(str(chunk["id"]).startswith(pdf_stem + "-"),
                            f"Chunk ID '{chunk['id']}' in {output_file.name} needs prefix '{pdf_stem}-'.")
            
            # Validate path segments based on specific test setup
            self.assertEqual(chunk.get("path_segment_0"), "Part_I", f"Chunk {chunk.get('id')} path_segment_0 mismatch")
            self.assertEqual(chunk.get("path_segment_1"), "Affiliate_Agreements", f"Chunk {chunk.get('id')} path_segment_1 mismatch")


    def test_all_configurations_speed_comparison(self):
        """Test PDF processing with various configurations (CPU/MPS, Sequential/Concurrent) and log speeds."""
        
        configurations = [
            {"name": "CPU Sequential", "device": AcceleratorDevice.CPU, "concurrency": False, "file": self.cpu_sequential_output_file, "requires_mps": False},
            {"name": "CPU Concurrent", "device": AcceleratorDevice.CPU, "concurrency": True, "file": self.cpu_concurrent_output_file, "requires_mps": False},
            {"name": "MPS Sequential", "device": AcceleratorDevice.MPS, "concurrency": False, "file": self.mps_sequential_output_file, "requires_mps": True},
            {"name": "MPS Concurrent", "device": AcceleratorDevice.MPS, "concurrency": True, "file": self.mps_concurrent_output_file, "requires_mps": True}
        ]

        results = []
        
        mps_available = False
        try:
            # Check if MPS is available via torch
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                mps_available = True
                logger.info(f"MPS accelerator: [bold green]Available[/]")
            else:
                logger.info("MPS accelerator: [bold red]Not available[/]")
        except Exception as e:
            logger.warning(f"Error checking MPS: [bold yellow]{e}[/]")

        rich_print("\n[bold cyan]Running Performance Tests[/]\n")

        for config in configurations:
            if config["requires_mps"] and not mps_available:
                logger.info(f"Skipping [bold yellow]{config['name']}[/] (MPS not available)")
                results.append({"name": config["name"], "time": "N/A (Skipped - MPS unavailable)"})
                continue

            rich_print(f"\n[bold]Running test: [cyan]{config['name']}[/][/]\n")
            start_time = time.perf_counter()
            
            # Note: The chunk_size and chunk_overlap in main.py's argparse defaults are 500 and 100 respectively.
            # The process_pdfs_in_directory function signature defaults are 1000 and 200.
            # The test was previously hardcoding 500 and 200.
            # For consistency with what `main.py --concurrent` would run if no chunk args are specified, 
            # it might be better to use the argparse defaults (500, 100) or make it configurable if testing other sizes.
            # Let's use 500 and 100 to align with main.py's CLI defaults when not specified.
            process_pdfs_in_directory(
                self.input_pdf_dir, 
                config["file"], 
                use_concurrency=config["concurrency"],
                accelerator_device=config["device"],
                chunk_size=500,  # Aligning with main.py CLI default
                chunk_overlap=100 # Aligning with main.py CLI default
            )
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            results.append({"name": config["name"], "time": duration})
            
            logger.info(f"[bold green]Test complete[/] in [bold cyan]{duration:.4f}[/] seconds")
            self._validate_csv_output(config["file"], self.source_pdf_names)

        # Print summary table
        rich_print("\n\n[bold cyan]Performance Results Summary[/]\n")
        
        # Find the fastest configuration
        fastest_time = float('inf')
        fastest_config = None
        
        for result in results:
            if isinstance(result["time"], float) and result["time"] < fastest_time:
                fastest_time = result["time"]
                fastest_config = result["name"]
        
        # Display results with highlighting for the fastest
        console.print("┌─────────────────────┬────────────────────┐")
        console.print("│ [bold]Configuration[/]      │ [bold]Execution Time[/]     │")
        console.print("├─────────────────────┼────────────────────┤")
        
        for result in results:
            name = result["name"]
            if isinstance(result["time"], float):
                time_str = f"{result['time']:.4f} seconds"
                # Add a star to indicate the fastest
                if name == fastest_config:
                    name = f"[bold green]{name} ⭐[/]"
                    time_str = f"[bold green]{time_str}[/]"
                console.print(f"│ {name:<19} │ {time_str:<18} │")
            else:
                console.print(f"│ {name:<19} │ {result['time']:<18} │")
        
        console.print("└─────────────────────┴────────────────────┘")
        
        if fastest_config:
            logger.info(f"[bold green]Fastest configuration: {fastest_config}[/] ({fastest_time:.4f} seconds)")


if __name__ == '__main__':
    # Use rich to print a nice welcome message
    rich_print("\n[bold cyan]DocuNomic PDF Processor - Test Suite[/]\n")

    # Set the multiprocessing start method to 'spawn' for consistency and safety
    try:
        current_method = multiprocessing.get_start_method(allow_none=True)
        if current_method != 'spawn':
            force_spawn = True if current_method else False
            multiprocessing.set_start_method('spawn', force=force_spawn)
            logger.info(f"Multiprocessing: [bold cyan]spawn[/] (was: {current_method})")
        else:
            logger.info(f"Multiprocessing: [bold cyan]spawn[/]")
    except RuntimeError as e:
        logger.warning(f"Could not set multiprocessing method: [bold yellow]{e}[/]")
    except Exception as e:
        logger.error(f"Unexpected error: [bold red]{e}[/]")

    unittest.main()