"""Shared preprocessing operations for the BIDS analysis pipeline."""

from __future__ import annotations

import warnings

import mne
import numpy as np
from mne_icalabel import label_components


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SRATE = 300
GROUP_IDS = list(range(1, 12))
N_SUBJ = 3
N_CH = 19
N_TRIALS = 40
REST_N_RUNS = 3

CH_NAMES = [
    "P3",
    "C3",
    "F3",
    "Fz",
    "F4",
    "C4",
    "P4",
    "Cz",
    "Pz",
    "Fp1",
    "Fp2",
    "T3",
    "T5",
    "O1",
    "O2",
    "F7",
    "F8",
    "T6",
    "T4",
]

MONTAGE = mne.channels.make_standard_montage("standard_1020")

ICA_N_COMPONENTS = 15
ICA_RANDOM_STATE = 42
ICA_MAX_ITER = 500
ICA_REJECT_LABELS = {"muscle artifact", "eye blink"}
ICA_REJECT_PROB = 0.80


def preprocess_task(eeg_trials: np.ndarray):
    """Average-reference, filter, and ICA-clean task EEG.

    Parameters
    ----------
    eeg_trials
        Array shaped `(n_channels, n_times, n_trials)` in microvolts.
    """
    n_ch, n_times, n_trials = eeg_trials.shape
    data_cat = eeg_trials.transpose(0, 2, 1).reshape(n_ch, -1) * 1e-6

    info = mne.create_info(CH_NAMES, sfreq=SRATE, ch_types="eeg")
    raw = mne.io.RawArray(data_cat, info, verbose=False)
    raw.set_montage(MONTAGE, on_missing="ignore", verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    raw.filter(1.0, 100.0, method="fir", verbose=False)

    ica = mne.preprocessing.ICA(
        n_components=min(ICA_N_COMPONENTS, n_ch - 1),
        method="infomax",
        fit_params=dict(extended=True),
        random_state=ICA_RANDOM_STATE,
        max_iter=ICA_MAX_ITER,
        verbose=False,
    )
    ica.fit(raw, verbose=False)

    ic = label_components(raw, ica, method="iclabel")
    excl = [
        i
        for i, (label, prob) in enumerate(zip(ic["labels"], ic["y_pred_proba"]))
        if label in ICA_REJECT_LABELS and prob > ICA_REJECT_PROB
    ]
    ica.exclude = excl
    ica.apply(raw, verbose=False)

    cleaned = raw.get_data().reshape(n_ch, n_trials, n_times).transpose(0, 2, 1)
    return cleaned * 1e6, len(excl)


def preprocess_resting(eeg_cont: np.ndarray):
    """Average-reference and 1-45 Hz filter resting EEG.

    Parameters
    ----------
    eeg_cont
        Array shaped `(n_channels, n_times)` in microvolts.
    """
    info = mne.create_info(CH_NAMES, sfreq=SRATE, ch_types="eeg")
    raw = mne.io.RawArray(eeg_cont * 1e-6, info, verbose=False)
    raw.set_montage(MONTAGE, on_missing="ignore", verbose=False)
    raw.set_eeg_reference("average", projection=False, verbose=False)
    raw.filter(1.0, 45.0, method="fir", verbose=False)
    return raw.get_data() * 1e6
