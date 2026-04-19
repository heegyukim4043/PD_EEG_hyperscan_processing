# Hyperscanning EEG — Prisoner's Dilemma

Analysis code for the three-person hyperscanning EEG dataset accompanying the paper:

> **"[Paper title]"**  
> [Authors] · [Journal] · [Year]

---

## Overview

Eleven groups of three participants simultaneously played a repeated prisoner's dilemma game (40 trials) while EEG was recorded with DSI-24 (19 channels per participant, 57 channels total per group).  
This repository provides the full analysis pipeline:

1. **Preprocessing** — ICA-based artifact removal (task) + bandpass filtering (resting)
2. **IBS analysis** — Inter-Brain Synchrony (PLV, Coherence) with cluster-permutation statistics
3. **ERP analysis** — Grand-average ERP, Coop vs. Defect comparison, difference waves

---

## Dataset

The raw `.mat` files are available at: **[Dataset DOI / Repository URL]**

Place the downloaded files in a `data/` folder:

```
data/
  G01_eeg.mat
  G02_eeg.mat
  ...
  G11_eeg.mat
```

### Data structure

Each `GXX_eeg.mat` file contains a MATLAB struct `data` with the following fields:

| Field | Shape | Description |
|---|---|---|
| `decision_X` | (57, 1500, 40) | Decision-phase EEG, epoch [−1000, 4000] ms, 300 Hz |
| `feedback_X` | (57, 900, 40) | Feedback-phase EEG, epoch [−1000, 2000] ms, 300 Hz |
| `resting` | (57, 18000, 3) | Resting-state EEG, 3 runs × 60 s, 300 Hz |
| `score` | (40, 3) | Behavioral scores per trial per subject (1 = cooperate, 2 = defect) |

- **57 channels = 19 ch × 3 subjects** (S1, S2, S3)
- Channel naming convention: `{electrode}` ordered as `P3, C3, F3, Fz, F4, C4, P4, Cz, Pz, Fp1, Fp2, T3, T5, O1, O2, F7, F8, T6, T4`; repeated for S1, S2, S3
- **Original recording filter** (applied prior to saving): high-pass 1 Hz · notch 60 Hz · reference: both earlobes · 300 Hz

### Experimental design

| Item | Detail |
|---|---|
| Groups (hyperscanning triads) | 11 (G01–G11) |
| Subjects per group | 3 |
| EEG channels per subject | 19 (DSI-24, 10–20 system) |
| Task | Repeated prisoner's dilemma |
| Trials | 40 per session |
| Decision epoch | [−1000, 4000] ms (onset at 0 ms) |
| Feedback epoch | [−1000, 2000] ms (onset at 0 ms) |
| Resting state | 3 runs × 60 s (eyes open) |
| Sampling rate | 300 Hz |
| Reference | Both earlobes |

---

## Installation

```bash
pip install -r requirements.txt
```

**Key dependencies:**

| Package | Purpose |
|---|---|
| `mne` | EEG processing, ICA, filtering |
| `mne-icalabel` | Automatic ICA component classification |
| `hypyp` | Inter-brain synchrony (PLV, Coherence, cluster-permutation) |
| `scipy` | Signal processing, statistics |
| `matplotlib` | Figures |

Python ≥ 3.9 recommended.

---

## Usage

Run the three scripts in order:

```bash
# Step 1 — Preprocessing (saves cleaned_eeg.pkl, ~20–40 min)
python preprocess.py --data_dir data/ --output_dir results/

# Step 2 — IBS analysis (saves figures + CSV, ~30–60 min for permutation tests)
python ibs_analysis.py --cache results/cleaned_eeg.pkl --output_dir results/

# Step 3 — ERP analysis (saves figures + CSV, ~2 min)
python erp_analysis.py --cache results/cleaned_eeg.pkl --output_dir results/
```

All scripts cache intermediate results. Re-running is safe and fast after the first run.

---

## Analysis pipeline

### 1. Preprocessing (`preprocess.py`)

**Task EEG (decision & feedback)**

| Step | Detail |
|---|---|
| Re-reference | Average reference |
| Bandpass | 1–100 Hz (FIR) |
| ICA | Infomax extended, n_components = min(15, n_ch − 1) = 15, seed = 42 |
| Artifact rejection | ICLabel: eye blink + muscle artifact (p > 0.80) |

**Resting-state EEG**

| Step | Detail |
|---|---|
| Re-reference | Average reference |
| Bandpass | 1–45 Hz (FIR) |
| ICA | Not applied |

**Output:** `results/cleaned_eeg.pkl`

```
cache[g]                          # g = 1 … 11
  ['decision'] : (57, 1500, 40)   # µV, ICA-cleaned
  ['feedback'] : (57,  900, 40)   # µV, ICA-cleaned
  ['resting']  : list of 3 runs, each (57, 18000)   # µV, filtered
  ['score']    : (40, 3)          # 1 = cooperate, 2 = defect
```

---

### 2. IBS analysis (`ibs_analysis.py`)

**Frequency bands**

| Band | Range |
|---|---|
| Delta | 1–3 Hz |
| Theta | 4–7 Hz |
| Alpha | 8–12 Hz |
| Beta | 14–25 Hz |
| Gamma | 30–45 Hz |

**Task IBS**
- Bandpass (Butterworth order 4) → Hilbert transform → PLV / Coherence (ccorr)
- Windows: 0–1000 ms and 0–2000 ms post-stimulus onset
- Subject pairs: S1–S2, S1–S3, S2–S3 (3 dyads per group)
- Condition: **Cooperative** (both subjects chose cooperate on that trial) vs. **Other**
- Feature vector per trial: 95 values = 5 bands × 19 channels (channel-wise IBS on matching pairs)

**Resting IBS**
- Non-overlapping windows, length matched to each task window (1000 ms or 2000 ms)

**Statistics**
- Cluster-permutation test (HyPyP `statscondCluster`)
- 2000 permutations, two-tailed (tail = 0), α = 0.05

**Outputs**

| File | Description |
|---|---|
| `ibs_data.pkl` | IBS arrays per task / window / metric / condition |
| `cluster_stats.pkl` | min_p, n_sig, F-statistic map per test |
| `stats_ibs_by_band.csv` | Per-band mean ± SEM, Cohen's d, cluster p |
| `fig_ibs_plv.png` | PLV bar figure (Coop vs. Other) |
| `fig_ibs_coh.png` | Coherence bar figure (Coop vs. Other) |
| `fig_fstat_*.png` | F-statistic heatmaps (band × channel) |

**`ibs_data.pkl` structure**

```
results['decision'][(0, 1000)]['plv']['coop']  : ndarray (n_coop_trials, 95)
results['decision'][(0, 1000)]['plv']['other'] : ndarray (n_other_trials, 95)
# keys: task ∈ {decision, feedback}
#       window ∈ {(0,1000), (0,2000)}
#       metric ∈ {plv, coh}
#       cond ∈ {coop, other}

results['resting'][1000]['plv']['all'] : ndarray (5940, 95)
results['resting'][2000]['plv']['all'] : ndarray (2970, 95)
# 1000 ms: 11 groups × 3 dyads × 3 runs × 60 windows = 5940
# 2000 ms: 11 groups × 3 dyads × 3 runs × 30 windows = 2970
```

---

### 3. ERP analysis (`erp_analysis.py`)

**Pipeline**
- Trial average per subject → low-pass 15 Hz (Butterworth, order 4) → baseline correction [−200, 0] ms
- Display window: −500 to 750 ms
- Channels of interest: Fz (idx 3), Cz (idx 7), Pz (idx 8)
- Minimum trials per condition for subject inclusion: 3

**Panels**

| Panel | Description |
|---|---|
| (A) Grand average | All trials, per channel |
| (B) Coop vs. Defect | Condition comparison, exploratory |
| (C) Difference wave | Coop − Defect ± SEM |
| (D) Group-level amplitude | Mean amplitude [250–500 ms] at Pz per group |

**ERP components annotated**

| Task | Components |
|---|---|
| Decision | N2 (~200 ms), P3 (~350 ms) |
| Feedback | FRN (~250 ms), P300 (~380 ms) |

**Outputs**

| File | Description |
|---|---|
| `fig_erp_decision.png` | Panels A + B, decision task |
| `fig_erp_feedback.png` | Panels A + B, feedback task |
| `fig_erp_diff.png` | Panel C — difference waves |
| `fig_erp_group.png` | Panel D — group-level amplitude |
| `erp_amplitude_summary.csv` | Mean amplitude per task / channel / window |

---

## Output directory structure

```
results/
  cleaned_eeg.pkl              preprocessed EEG cache
  ibs_data.pkl                 IBS arrays
  cluster_stats.pkl            cluster-permutation results
  stats_ibs_by_band.csv        IBS summary table
  erp_amplitude_summary.csv    ERP amplitude table
  fig_ibs_plv.png
  fig_ibs_coh.png
  fig_fstat_decision_plv.png
  fig_fstat_decision_coh.png
  fig_fstat_feedback_plv.png
  fig_fstat_feedback_coh.png
  fig_erp_decision.png
  fig_erp_feedback.png
  fig_erp_diff.png
  fig_erp_group.png
```

---

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{,
  title   = {},
  author  = {},
  journal = {},
  year    = {},
  doi     = {}
}
```

---

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
