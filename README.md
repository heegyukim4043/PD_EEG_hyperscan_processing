# BIDS-Based PD EEG Analysis

This folder contains analysis code for the EEG-BIDS dataset in `../eeg_bids`.

The code does not require MATLAB `.mat` files. It reads:

- EEGLAB `.set` EEG files
- BIDS `events.tsv`
- BIDS participant/task metadata

## Scripts

| Script | Purpose |
|---|---|
| `preprocess_bids.py` | Reads BIDS EEG files and creates the cleaned analysis cache |
| `preprocessing_core.py` | Shared MNE preprocessing functions |
| `ibs_analysis.py` | IBS analysis: PLV, coherence, cluster statistics |
| `erp_analysis.py` | ERP analysis and figures |
| `run_all.py` | Runs all steps in order |

## Install

From the project root:

```powershell
.\.venv\Scripts\python -m pip install -r pd_eeg_analysis\requirements.txt
```

## Run Full Pipeline

```powershell
.\.venv\Scripts\python pd_eeg_analysis\run_all.py
```

Default paths:

```text
Input BIDS dataset: eeg_bids/
Output folder:      results/
```

## Run Individual Steps

Preprocessing:

```powershell
.\.venv\Scripts\python pd_eeg_analysis\preprocess_bids.py --bids_dir eeg_bids --output_dir results
```

IBS analysis:

```powershell
.\.venv\Scripts\python pd_eeg_analysis\ibs_analysis.py --cache results\cleaned_eeg.pkl --output_dir results
```

ERP analysis:

```powershell
.\.venv\Scripts\python pd_eeg_analysis\erp_analysis.py --cache results\cleaned_eeg.pkl --output_dir results
```

## Runtime

`preprocess_bids.py` runs ICA for task EEG and can take a long time.

`ibs_analysis.py` computes synchrony features and 2000-permutation cluster tests, so it can also take a long time.
