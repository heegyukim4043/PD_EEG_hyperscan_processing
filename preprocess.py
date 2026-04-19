"""
preprocess.py
=============
Preprocessing pipeline for three-person hyperscanning EEG.

Input
-----
data/<GXX>_eeg.mat  (one file per group, G01–G11)
  data.decision_X : (57, 1500, 40)   µV, 300 Hz, epoch [-1000, 4000] ms
  data.feedback_X : (57,  900, 40)   µV, 300 Hz, epoch [-1000, 2000] ms
  data.resting    : (57, 18000, 3)   µV, 300 Hz, 3 × 60 s runs
  data.score      : (40, 3)          1 = cooperate, 2 = defect

  57 channels = 19 ch × 3 subjects (S1, S2, S3)
  Original recording filter applied in .mat:
    high-pass 1 Hz | notch 60 Hz | reference: both earlobes

Task EEG pipeline
-----------------
  Average reference → bandpass 1–100 Hz (FIR) → ICA (infomax extended, 15 comp)
  → ICLabel rejection (eye blink + muscle artifact, p > 0.80)

Resting EEG pipeline
--------------------
  Average reference → bandpass 1–45 Hz (FIR)   [no ICA for speed]

Output
------
results/cleaned_eeg.pkl
  cache[g]['decision']  : ndarray (57, n_times, 40)
  cache[g]['feedback']  : ndarray (57, n_times, 40)
  cache[g]['resting']   : list of 3 × ndarray (57, 18000)
  cache[g]['score']     : ndarray (40, 3)

Usage
-----
  python preprocess.py
  python preprocess.py --data_dir path/to/data --output_dir path/to/results
"""

import os
import argparse
import pickle
import warnings

import numpy as np
import scipy.io as sio
import mne
from mne_icalabel import label_components

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ── Default paths ──────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.join(BASE_DIR, '..', 'data')
_OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'results')

# ── Recording parameters ───────────────────────────────────────────────────────
SRATE     = 300
GROUP_IDS = list(range(1, 12))
N_SUBJ    = 3
N_CH      = 19
N_TRIALS  = 40

CH_NAMES = ['P3','C3','F3','Fz','F4','C4','P4','Cz','Pz',
            'Fp1','Fp2','T3','T5','O1','O2','F7','F8','T6','T4']

MONTAGE = mne.channels.make_standard_montage('standard_1020')

# ── ICA parameters ─────────────────────────────────────────────────────────────
ICA_N_COMPONENTS  = 15          # min(15, n_ch - 1) = 15 for 19-ch data
ICA_RANDOM_STATE  = 42
ICA_MAX_ITER      = 500
ICA_REJECT_LABELS = {'muscle artifact', 'eye blink'}
ICA_REJECT_PROB   = 0.80

# ── Resting state ──────────────────────────────────────────────────────────────
REST_N_RUNS = 3


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════
def _get_field(data, field):
    val = data[field]
    return val.item() if hasattr(val, 'item') else val


def load_group(g, data_dir):
    mat = sio.loadmat(os.path.join(data_dir, f'G{g:02d}_eeg.mat'),
                      squeeze_me=True)
    d = mat['data']
    return {
        'decision_X': _get_field(d, 'decision_X'),
        'feedback_X': _get_field(d, 'feedback_X'),
        'resting':    _get_field(d, 'resting'),
        'score':      _get_field(d, 'score'),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Task EEG: ICA-based artifact removal
# ══════════════════════════════════════════════════════════════════════════════
def preprocess_task(eeg_trials):
    """ICA artifact removal for task EEG (epochs concatenated then split back).

    Parameters
    ----------
    eeg_trials : ndarray (n_ch, n_times, n_trials)  µV

    Returns
    -------
    cleaned : ndarray (n_ch, n_times, n_trials)  µV
    n_rejected : int  number of ICA components rejected
    """
    n_ch, n_times, n_trials = eeg_trials.shape
    data_cat = eeg_trials.transpose(0, 2, 1).reshape(n_ch, -1) * 1e-6

    info = mne.create_info(CH_NAMES, sfreq=SRATE, ch_types='eeg')
    raw  = mne.io.RawArray(data_cat, info, verbose=False)
    raw.set_montage(MONTAGE, on_missing='ignore', verbose=False)
    raw.set_eeg_reference('average', projection=False, verbose=False)
    raw.filter(1., 100., method='fir', verbose=False)

    ica = mne.preprocessing.ICA(
        n_components=min(ICA_N_COMPONENTS, n_ch - 1),
        method='infomax',
        fit_params=dict(extended=True),
        random_state=ICA_RANDOM_STATE,
        max_iter=ICA_MAX_ITER,
        verbose=False,
    )
    ica.fit(raw, verbose=False)

    ic   = label_components(raw, ica, method='iclabel')
    excl = [i for i, (lbl, prob) in
            enumerate(zip(ic['labels'], ic['y_pred_proba']))
            if lbl in ICA_REJECT_LABELS and prob > ICA_REJECT_PROB]
    ica.exclude = excl
    ica.apply(raw, verbose=False)

    cleaned = raw.get_data().reshape(n_ch, n_trials, n_times).transpose(0, 2, 1)
    return cleaned * 1e6, len(excl)


# ══════════════════════════════════════════════════════════════════════════════
# Resting EEG: bandpass filter only
# ══════════════════════════════════════════════════════════════════════════════
def preprocess_resting(eeg_cont):
    """Bandpass filter and average reference for resting-state EEG.

    Parameters
    ----------
    eeg_cont : ndarray (n_ch, n_times)  µV

    Returns
    -------
    filtered : ndarray (n_ch, n_times)  µV
    """
    info = mne.create_info(CH_NAMES, sfreq=SRATE, ch_types='eeg')
    raw  = mne.io.RawArray(eeg_cont * 1e-6, info, verbose=False)
    raw.set_montage(MONTAGE, on_missing='ignore', verbose=False)
    raw.set_eeg_reference('average', projection=False, verbose=False)
    raw.filter(1., 45., method='fir', verbose=False)
    return raw.get_data() * 1e6


# ══════════════════════════════════════════════════════════════════════════════
# Main preprocessing loop
# ══════════════════════════════════════════════════════════════════════════════
def run_preprocessing(data_dir, output_dir):
    cache_path = os.path.join(output_dir, 'cleaned_eeg.pkl')
    if os.path.exists(cache_path):
        print(f'Cache found at {cache_path} — loading.')
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    cache = {}
    for g in GROUP_IDS:
        print(f'\n[G{g:02d}]', flush=True)
        grp = load_group(g, data_dir)
        cache[g] = {'score': grp['score'], 'resting': []}

        for task, key in [('decision', 'decision_X'), ('feedback', 'feedback_X')]:
            print(f'  {task}:', end='', flush=True)
            eeg_raw = grp[key]                         # (57, n_times, 40)
            cleaned = np.zeros_like(eeg_raw, dtype=float)
            total_rej = 0
            for s in range(N_SUBJ):
                sl = slice(s * N_CH, (s + 1) * N_CH)
                cleaned[sl], n_rej = preprocess_task(eeg_raw[sl])
                total_rej += n_rej
                print(f'  S{s+1}({n_rej}rej)', end='', flush=True)
            cache[g][task] = cleaned
            print()

        print('  resting:', end='', flush=True)
        for run_idx in range(REST_N_RUNS):
            rest_run     = grp['resting'][:, :, run_idx]
            cleaned_rest = np.zeros_like(rest_run, dtype=float)
            for s in range(N_SUBJ):
                sl = slice(s * N_CH, (s + 1) * N_CH)
                cleaned_rest[sl] = preprocess_resting(rest_run[sl])
            cache[g]['resting'].append(cleaned_rest)
            print(f' run{run_idx+1}', end='', flush=True)
        print()

    os.makedirs(output_dir, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f)
    print(f'\nSaved: {cache_path}')
    return cache


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description='Hyperscanning EEG preprocessing')
    p.add_argument('--data_dir',   default=_DATA_DIR,
                   help='Directory containing G01_eeg.mat … G11_eeg.mat')
    p.add_argument('--output_dir', default=_OUTPUT_DIR,
                   help='Directory to write cleaned_eeg.pkl')
    return p.parse_args()


def main():
    args = parse_args()
    print('=== Hyperscanning EEG Preprocessing ===')
    print(f'  data_dir   : {args.data_dir}')
    print(f'  output_dir : {args.output_dir}')
    run_preprocessing(args.data_dir, args.output_dir)
    print('\nDone. Run ibs_analysis.py next.')


if __name__ == '__main__':
    main()
