# docunomic

![docunomic](public/docunomic.png)

Document processing for Nomic

## Demo

#### How do you get this data into Atlas?

```bash
├── Part_I
│   ├── Affiliate_Agreements
│   │   ├── CreditcardscomInc_20070810_S-1_EX-10.33_362297_EX-10.33_Affiliate Agreement.pdf
│   │   ├── CybergyHoldingsInc_20140520_10-Q_EX-10.27_8605784_EX-10.27_Affiliate Agreement.pdf
│   │   ├── DigitalCinemaDestinationsCorp_20111220_S-1_EX-10.10_7346719_EX-10.10_Affiliate Agreement.pdf
│   │   ├── LinkPlusCorp_20050802_8-K_EX-10_3240252_EX-10_Affiliate Agreement.pdf
│   │   ├── SouthernStarEnergyInc_20051202_SB-2A_EX-9_801890_EX-9_Affiliate Agreement.pdf
│   │   ├── SteelVaultCorp_20081224_10-K_EX-10.16_3074935_EX-10.16_Affiliate Agreement.pdf
│   │   ├── TubeMediaCorp_20060310_8-K_EX-10.1_513921_EX-10.1_Affiliate Agreement.pdf
│   │   ├── UnionDentalHoldingsInc_20050204_8-KA_EX-10_3345577_EX-10_Affiliate Agreement.pdf
│   │   └── UsioInc_20040428_SB-2_EX-10.11_1723988_EX-10.11_Affiliate Agreement 2.pdf
│   ├── Co_Branding
│   │   ├── 2ThemartComInc_19990826_10-12G_EX-10.10_6700288_EX-10.10_Co-Branding 
...
...
...
    └── Transportation
        ├── ENERGYXXILTD_05_08_2015-EX-10.13-Transportation AGREEMENT.PDF
        ├── ENTERPRISEPRODUCTSPARTNERSLP_07_08_1998-EX-10.3-TRANSPORTATION CONTRACT.PDF
        └── MARTINMIDSTREAMPARTNERSLP_01_23_2004-EX-10.3-TRANSPORTATION SERVICES AGREEMENT.PDF

64 directories, 510 files
```

#### Run docunomic!

```bash
uv run main.py input_dir data/CUAD_v1/full_contract_pdf output_file cuad.csv
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