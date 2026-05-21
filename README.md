# PD EEG Hyperscanning Processing

Python analysis pipeline for the three-person hyperscanning EEG prisoner's dilemma dataset.

This folder is adapted from:

https://github.com/heegyukim4043/PD_EEG_hyperscan_processing

The analysis logic is kept the same as the source repository. The local changes are:

- input `.mat` files are read from the parent project folder by default
- outputs are written to `results/` by default
- `run_all.py` was added to run preprocessing, IBS, and ERP steps in order
- MAT struct loading was adjusted for the current `GXX_eeg.mat` files

## Expected Data

Place the group-level MATLAB files in the project root, next to this `pd_eeg_analysis/` folder:

```text
project_root/
  G01_eeg.mat
  G02_eeg.mat
  ...
  G11_eeg.mat
  pd_eeg_analysis/
    preprocess.py
    ibs_analysis.py
    erp_analysis.py
    run_all.py
```

Each `GXX_eeg.mat` file must contain a MATLAB struct named `data`:

| Field | Shape | Description |
|---|---:|---|
| `decision_X` | `(57, 1500, 40)` | Decision-phase EEG, epoch `[-1000, 4000]` ms, 300 Hz |
| `feedback_X` | `(57, 900, 40)` | Feedback-phase EEG, epoch `[-1000, 2000]` ms, 300 Hz |
| `resting` | `(57, 18000, 3)` | Resting-state EEG, 3 runs x 60 s, 300 Hz |
| `score` | `(40, 3)` | Trial behavior per subject, `1 = cooperate`, `2 = defect` |

Channel order per subject:

```text
P3, C3, F3, Fz, F4, C4, P4, Cz, Pz, Fp1, Fp2, T3, T5, O1, O2, F7, F8, T6, T4
```

The 57 channels are stored as `19 channels x 3 subjects`:

- S1: channels 1-19
- S2: channels 20-38
- S3: channels 39-57

## Installation

Create or activate a Python environment, then install dependencies:

```powershell
.\.venv\Scripts\python -m pip install -r pd_eeg_analysis\requirements.txt
```

Requirements:

- Python 3.9 or newer
- `mne`
- `mne-icalabel`
- `hypyp`
- `numpy`
- `scipy`
- `matplotlib`

## Run All Steps

From the project root:

```powershell
.\.venv\Scripts\python pd_eeg_analysis\run_all.py
```

This runs:

1. `preprocess.py`
2. `ibs_analysis.py`
3. `erp_analysis.py`

Outputs are written to:

```text
results/
```

## Run Individual Steps

Preprocessing:

```powershell
.\.venv\Scripts\python pd_eeg_analysis\preprocess.py --data_dir . --output_dir results
```

IBS analysis:

```powershell
.\.venv\Scripts\python pd_eeg_analysis\ibs_analysis.py --cache results\cleaned_eeg.pkl --output_dir results
```

ERP analysis:

```powershell
.\.venv\Scripts\python pd_eeg_analysis\erp_analysis.py --cache results\cleaned_eeg.pkl --output_dir results
```

To skip expensive steps:

```powershell
.\.venv\Scripts\python pd_eeg_analysis\run_all.py --skip-preprocess
.\.venv\Scripts\python pd_eeg_analysis\run_all.py --skip-ibs
.\.venv\Scripts\python pd_eeg_analysis\run_all.py --skip-erp
```

## Pipeline

### 1. Preprocessing

Task EEG:

- average reference
- 1-100 Hz FIR bandpass
- ICA with extended Infomax
- ICLabel rejection for eye blink and muscle artifact components with probability `> 0.80`

Resting EEG:

- average reference
- 1-45 Hz FIR bandpass
- no ICA

Main output:

```text
results/cleaned_eeg.pkl
```

Cache structure:

```python
cache[g]["decision"]  # ndarray, shape (57, 1500, 40)
cache[g]["feedback"]  # ndarray, shape (57, 900, 40)
cache[g]["resting"]   # list of 3 arrays, each shape (57, 18000)
cache[g]["score"]     # ndarray, shape (40, 3)
```

### 2. IBS Analysis

Metrics:

- PLV
- coherence using HyPyP `ccorr`

Frequency bands:

| Band | Range |
|---|---:|
| Delta | 1-3 Hz |
| Theta | 4-7 Hz |
| Alpha | 8-12 Hz |
| Beta | 14-25 Hz |
| Gamma | 30-45 Hz |

Task windows:

- 0-1000 ms
- 0-2000 ms

Dyads:

- S1-S2
- S1-S3
- S2-S3

Condition split:

- cooperative: both subjects chose cooperate on the trial
- other: all remaining trials

Statistics:

- HyPyP cluster-permutation test
- 2000 permutations
- two-tailed
- alpha = 0.05

Main outputs:

```text
results/ibs_data.pkl
results/cluster_stats.pkl
results/stats_ibs_by_band.csv
results/fig_ibs_plv.png
results/fig_ibs_coh.png
```

## 3. ERP Analysis

ERP steps:

- average trials per subject
- 15 Hz low-pass Butterworth filter
- baseline correction using `[-200, 0]` ms
- plot window `[-500, 750]` ms

Channels of interest:

- Fz
- Cz
- Pz

Main outputs:

```text
results/fig_erp_decision.png
results/fig_erp_feedback.png
results/fig_erp_diff.png
results/fig_erp_group.png
results/erp_amplitude_summary.csv
```

## Runtime Notes

`preprocess.py` can take a long time because it runs ICA for each subject and task.

`ibs_analysis.py` can also take a long time because it computes synchrony features and runs 2000-permutation cluster tests.

For quick checking after preprocessing, ERP can be run independently:

```powershell
.\.venv\Scripts\python pd_eeg_analysis\run_all.py --skip-preprocess --skip-ibs
```
