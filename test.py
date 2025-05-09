import unittest
import tempfile
import shutil
# import json # No longer needed directly in this test after changes
import csv
from pathlib import Path
import time # Added for timing
import os # For listing files if needed, though Path.glob should suffice
import multiprocessing # Added for setting start method

# Assuming run.py is in the same directory or accessible in PYTHONPATH
from main import process_pdfs_in_directory 
from docling.datamodel.pipeline_options import AcceleratorDevice # Added import

class TestPdfProcessing(unittest.TestCase):

    def setUp(self):
        """Set up temporary directories for test input and output."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.input_pdf_dir = self.test_dir / "input_pdfs"
        self.input_pdf_dir.mkdir()

        # Create the expected subdirectory structure for testing path segment extraction
        self.structured_input_dir = self.input_pdf_dir / "Part_I" / "Affiliate_Agreements"
        self.structured_input_dir.mkdir(parents=True, exist_ok=True)
        
        # Output files for different configurations
        self.cpu_sequential_output_file = self.test_dir / "output_chunks_cpu_sequential.csv"
        self.cpu_concurrent_output_file = self.test_dir / "output_chunks_cpu_concurrent.csv"
        self.mps_sequential_output_file = self.test_dir / "output_chunks_mps_sequential.csv"
        self.mps_concurrent_output_file = self.test_dir / "output_chunks_mps_concurrent.csv"

        # Directory containing all source PDFs for the test
        self.source_agreements_dir = Path("data/CUAD_v1/full_contract_pdf/Part_I/Affiliate_Agreements/")

        if not self.source_agreements_dir.is_dir():
            raise FileNotFoundError(f"Source PDF directory not found: {self.source_agreements_dir}")

        self.source_pdf_names = set()
        copied_pdf_count = 0

        for source_pdf_path in self.source_agreements_dir.glob("*.pdf"):
            if source_pdf_path.is_file():
                # Copy PDFs into the structured directory
                target_pdf_path = self.structured_input_dir / source_pdf_path.name
                shutil.copy(source_pdf_path, target_pdf_path)
                self.source_pdf_names.add(source_pdf_path.name)
                copied_pdf_count += 1
        
        if copied_pdf_count == 0:
            raise FileNotFoundError(f"No PDF files found in source directory: {self.source_agreements_dir}")
        
        print(f"Set up test with {copied_pdf_count} PDF files from {self.source_agreements_dir}")


    def tearDown(self):
        """Remove temporary directories after the test."""
        shutil.rmtree(self.test_dir)

    def _validate_csv_output(self, output_file: Path, expected_pdf_names: set):
        """Helper function to validate CSV output."""
        self.assertTrue(output_file.exists(), f"Output CSV file {output_file.name} was not created.")
        self.assertTrue(output_file.is_file(), f"Output path {output_file.name} is not a file.")

        chunks_data = []
        try:
            with open(output_file, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("page_number") and row["page_number"].isdigit():
                        row["page_number"] = int(row["page_number"])
                    elif row.get("page_number") == '' or row.get("page_number") is None:
                        row["page_number"] = None
                    chunks_data.append(row)
        except FileNotFoundError:
            self.fail(f"Output CSV file {output_file.name} not found during read.")
        except Exception as e:
            self.fail(f"Error reading or parsing CSV file {output_file.name}: {e}")
        
        self.assertTrue(len(chunks_data) > 0, f"No chunks were produced in {output_file.name}.")

        processed_pdf_filenames = {chunk["source_pdf_filename"] for chunk in chunks_data}
        self.assertEqual(expected_pdf_names, processed_pdf_filenames,
                         f"Processed PDF filenames in {output_file.name} do not match expected. "
                         f"Expected: {expected_pdf_names}, Got: {processed_pdf_filenames}")

        expected_headers = ["id", "text", "source_pdf_filename", "heading", "page_number", 
                            "document_title", "part", "category", "docling_origin_filename"]
        for chunk in chunks_data: # Check headers for each chunk for robustness
            for header in expected_headers:
                self.assertIn(header, chunk, f"CSV data in {output_file.name} missing '{header}' field in a chunk.")
            
            if chunk.get("page_number") is not None:
                 self.assertIsInstance(chunk["page_number"], int, f"'page_number' in {output_file.name} should be an int or None after conversion.")

            # Check ID format: filename_stem-chunk_index
            pdf_stem = Path(chunk["source_pdf_filename"]).stem
            self.assertTrue(chunk["id"].startswith(pdf_stem + "-"),
                            f"Chunk ID '{chunk['id']}' in {output_file.name} does not start with expected prefix '{pdf_stem}-'.")


    def test_all_configurations_speed_comparison(self):
        """Test processing multiple PDFs with different configurations (CPU/MPS, Seq/Conc) and compare speeds."""
        
        configurations = [
            {
                "name": "CPU Sequential", 
                "device": AcceleratorDevice.CPU, 
                "concurrency": False, 
                "file": self.cpu_sequential_output_file,
                "skip_mps_on_non_mac": False
            },
            {
                "name": "CPU Concurrent", 
                "device": AcceleratorDevice.CPU, 
                "concurrency": True, 
                "file": self.cpu_concurrent_output_file,
                "skip_mps_on_non_mac": False
            },
            {
                "name": "MPS Sequential", 
                "device": AcceleratorDevice.MPS, 
                "concurrency": False, 
                "file": self.mps_sequential_output_file,
                "skip_mps_on_non_mac": True
            },
            {
                "name": "MPS Concurrent", 
                "device": AcceleratorDevice.MPS, 
                "concurrency": True, 
                "file": self.mps_concurrent_output_file,
                "skip_mps_on_non_mac": True
            }
        ]

        results = []
        
        # Check if MPS is available (very basic check, assumes Mac for MPS)
        # A more robust check would involve trying to use torch.backends.mps.is_available()
        # but that introduces a torch dependency here which might not be desired.
        # For now, we'll assume MPS might not work on non-Darwin systems and allow skipping.
        is_macos = (os.uname().sysname == "Darwin")


        for config in configurations:
            if config["skip_mps_on_non_mac"] and not is_macos:
                print(f"\nSkipping {config['name']} as MPS is typically for macOS and system is not Darwin.")
                results.append({"name": config["name"], "time": "N/A (Skipped)"})
                continue

            print(f"\n--- Starting test: {config['name']} ---")
            start_time = time.perf_counter()
            
            process_pdfs_in_directory(
                self.input_pdf_dir, 
                config["file"], 
                use_concurrency=config["concurrency"],
                accelerator_device=config["device"]
            )
            
            end_time = time.perf_counter()
            duration = end_time - start_time
            results.append({"name": config["name"], "time": duration})
            
            print(f"--- Finished test: {config['name']} in {duration:.4f} seconds ---")
            self._validate_csv_output(config["file"], self.source_pdf_names)

        print("\n\n--- Test Execution Summary ---")
        for result in results:
            if isinstance(result["time"], float):
                print(f"{result['name']}: {result['time']:.4f} seconds")
            else:
                print(f"{result['name']}: {result['time']}")
        
        print("\n--- End of Summary ---")
        
        # You could add assertions here to compare times, but as noted,
        # this can be flaky. The printed summary is the main goal.
        # For example, to check if CPU concurrent was faster than CPU sequential:
        # cpu_seq_time = next((r['time'] for r in results if r['name'] == "CPU Sequential"), float('inf'))
        # cpu_conc_time = next((r['time'] for r in results if r['name'] == "CPU Concurrent"), float('inf'))
        # if isinstance(cpu_seq_time, float) and isinstance(cpu_conc_time, float) and self.source_pdf_names: # only if times are numbers and there were files
        #     print(f"CPU Sequential vs Concurrent: {cpu_seq_time:.2f}s vs {cpu_conc_time:.2f}s")
        #     if cpu_conc_time < cpu_seq_time:
        #         print(f"CPU Concurrent was {cpu_seq_time/cpu_conc_time:.2f}x faster.")
        #     else:
        #         print(f"CPU Concurrent was {cpu_conc_time/cpu_seq_time:.2f}x slower or equal.")
        # Similar comparisons can be done for MPS if MPS tests ran.


if __name__ == '__main__':
    # Set the start method to 'spawn' for cleaner process creation,
    # especially important when using GPU resources like MPS with multiprocessing.
    # This should be done before any multiprocessing Pool or Executor is created.
    # The 'force=True' might be needed if the context could be initialized by a library beforehand,
    # though typically it's used if set_start_method might be called multiple times with the same method.
    try:
        multiprocessing.set_start_method('spawn') # Using 'spawn' for better isolation
    except RuntimeError as e:
        # This can happen if the start method has already been set and we try to set it again,
        # or if context is already used. Check if it's already 'spawn'.
        if multiprocessing.get_start_method(allow_none=True) != 'spawn':
            print(f"Warning: Could not set multiprocessing start method to spawn: {e}. Current method: {multiprocessing.get_start_method(allow_none=True)}")
        # If it is already spawn, it's fine. If not, we print a warning but proceed.
        pass 

    unittest.main()