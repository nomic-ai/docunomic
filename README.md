# docunomic

![docunomic](public/docunomic.png)

Document processing for Nomic

## Run demo

```bash
uv run main.py pdf_input_dir data/CUAD_v1/full_contract_pdf output_file cuad.csv
```

This produces a CSV ready for Atlas
| id | text | source_pdf_filename | heading | page_number | document_title | part | category | docling_origin_filename |
|-----|------|-------------------|---------|-------------|----------------|------|----------|------------------------|
| OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-A... | May 21, 2015 Tribute Pharmaceutica... | OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-A... | | 1 | OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-A... | Part_II | Agency Agreements | OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-A... |
| OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-A... | Dear Mr. Harris: The undersigned, Du... | OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-A... | Attention: Rob Harris, President and Chief Executive Officer| 1| OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-A... | Part_II | Agency Agreements | OLDAPIWIND-DOWNLTD_01_08_2016-EX-1.3-A... |
...
## Diagnostic test

This test processes the 9 pdfs in `CUAD_v1/full_contract_pdf/Part_1/Affiliate_Agreements/` to test the performance of hardware and processing configuration

```bash
uv run test.py
```

Results on my nomic macbook m3 16gb:

```bash
--- Test Execution Summary ---
CPU Sequential: 45.5142 seconds
CPU Concurrent: 42.1427 seconds
MPS Sequential: 24.6420 seconds
MPS Concurrent: 38.2389 seconds
```