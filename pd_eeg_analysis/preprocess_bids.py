"""Preprocess EEG-BIDS EEGLAB files into the analysis cache.

This is the BIDS-only entry point for OpenNeuro-style datasets. It reads
participant-level EEGLAB `.set` files and `events.tsv` files, reconstructs the
group-level arrays expected by the original IBS and ERP analysis scripts, then
applies the same preprocessing operations as `preprocess.py`.
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from preprocessing_core import (
    GROUP_IDS,
    N_CH,
    N_SUBJ,
    REST_N_RUNS,
    preprocess_resting,
    preprocess_task,
)


ROOT = Path(__file__).resolve().parent.parent
_BIDS_DIR = ROOT / "eeg_bids"
_OUTPUT_DIR = ROOT / "results"
SUBJECT_SLOTS = ("S01", "S02", "S03")
TASK_FILES = {
    "decision": "pddecision",
    "feedback": "pdfeedback",
    "resting": "pdrest",
}


def _read_epochs_set(path: Path) -> np.ndarray:
    epochs = mne.io.read_epochs_eeglab(path, verbose=False)
    data = epochs.get_data(copy=True)
    return np.transpose(data, (1, 2, 0))


def _read_rest_set(path: Path) -> list[np.ndarray]:
    raw = mne.io.read_raw_eeglab(path, preload=True, verbose=False)
    data = raw.get_data()
    run_len = data.shape[1] // REST_N_RUNS
    return [data[:, idx * run_len : (idx + 1) * run_len] for idx in range(REST_N_RUNS)]


def _subject_path(bids_dir: Path, group_id: int, subject_slot: str, task_label: str) -> Path:
    sub = f"G{group_id:02d}{subject_slot}"
    return bids_dir / f"sub-{sub}" / "eeg" / f"sub-{sub}_task-{task_label}_eeg.set"


def _events_path(bids_dir: Path, group_id: int, subject_slot: str, task_label: str) -> Path:
    sub = f"G{group_id:02d}{subject_slot}"
    return bids_dir / f"sub-{sub}" / "eeg" / f"sub-{sub}_task-{task_label}_events.tsv"


def _load_score(bids_dir: Path, group_id: int) -> np.ndarray:
    score = np.zeros((40, N_SUBJ), dtype=int)
    for si, subject_slot in enumerate(SUBJECT_SLOTS):
        events = pd.read_csv(_events_path(bids_dir, group_id, subject_slot, TASK_FILES["decision"]), sep="\t")
        if "choice" not in events:
            raise ValueError(f"Missing choice column in events file for G{group_id:02d}{subject_slot}")
        choices = pd.to_numeric(events["choice"], errors="coerce").fillna(0).astype(int).to_numpy()
        score[: len(choices), si] = choices
    return score


def load_group_from_bids(group_id: int, bids_dir: Path) -> dict:
    group = {
        "decision_X": np.zeros((N_CH * N_SUBJ, 1500, 40), dtype=float),
        "feedback_X": np.zeros((N_CH * N_SUBJ, 900, 40), dtype=float),
        "resting_runs": [np.zeros((N_CH * N_SUBJ, 18000), dtype=float) for _ in range(REST_N_RUNS)],
        "score": _load_score(bids_dir, group_id),
    }

    for si, subject_slot in enumerate(SUBJECT_SLOTS):
        ch_slice = slice(si * N_CH, (si + 1) * N_CH)
        group["decision_X"][ch_slice] = _read_epochs_set(
            _subject_path(bids_dir, group_id, subject_slot, TASK_FILES["decision"])
        )
        group["feedback_X"][ch_slice] = _read_epochs_set(
            _subject_path(bids_dir, group_id, subject_slot, TASK_FILES["feedback"])
        )
        for run_idx, rest_run in enumerate(
            _read_rest_set(_subject_path(bids_dir, group_id, subject_slot, TASK_FILES["resting"]))
        ):
            group["resting_runs"][run_idx][ch_slice] = rest_run
    return group


def run_preprocessing_bids(bids_dir: str | Path, output_dir: str | Path):
    bids_dir = Path(bids_dir)
    output_dir = Path(output_dir)
    cache_path = output_dir / "cleaned_eeg.pkl"
    if cache_path.exists():
        print(f"Cache found at {cache_path} - loading.")
        with cache_path.open("rb") as f:
            return pickle.load(f)

    cache = {}
    for group_id in GROUP_IDS:
        print(f"\n[G{group_id:02d}]", flush=True)
        grp = load_group_from_bids(group_id, bids_dir)
        cache[group_id] = {"score": grp["score"], "resting": []}

        for task, key in [("decision", "decision_X"), ("feedback", "feedback_X")]:
            print(f"  {task}:", end="", flush=True)
            eeg_raw = grp[key]
            cleaned = np.zeros_like(eeg_raw, dtype=float)
            total_rej = 0
            for subject_idx in range(N_SUBJ):
                ch_slice = slice(subject_idx * N_CH, (subject_idx + 1) * N_CH)
                cleaned[ch_slice], n_rej = preprocess_task(eeg_raw[ch_slice])
                total_rej += n_rej
                print(f"  S{subject_idx + 1}({n_rej}rej)", end="", flush=True)
            cache[group_id][task] = cleaned
            print(f"  total={total_rej}")

        print("  resting:", end="", flush=True)
        for run_idx, rest_run in enumerate(grp["resting_runs"]):
            cleaned_rest = np.zeros_like(rest_run, dtype=float)
            for subject_idx in range(N_SUBJ):
                ch_slice = slice(subject_idx * N_CH, (subject_idx + 1) * N_CH)
                cleaned_rest[ch_slice] = preprocess_resting(rest_run[ch_slice])
            cache[group_id]["resting"].append(cleaned_rest)
            print(f" run{run_idx + 1}", end="", flush=True)
        print()

    output_dir.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
        pickle.dump(cache, f)
    print(f"\nSaved: {cache_path}")
    return cache


def parse_args():
    parser = argparse.ArgumentParser(description="BIDS-only hyperscanning EEG preprocessing")
    parser.add_argument("--bids_dir", default=str(_BIDS_DIR), help="EEG-BIDS dataset directory")
    parser.add_argument("--output_dir", default=str(_OUTPUT_DIR), help="Directory to write cleaned_eeg.pkl")
    return parser.parse_args()


def main():
    args = parse_args()
    print("=== BIDS EEG Preprocessing ===")
    print(f"  bids_dir   : {args.bids_dir}")
    print(f"  output_dir : {args.output_dir}")
    run_preprocessing_bids(args.bids_dir, args.output_dir)
    print("\nDone. Run ibs_analysis.py and erp_analysis.py next.")


if __name__ == "__main__":
    main()
