"""Run the original PD hyperscanning EEG pipeline on this folder's MAT files."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PIPELINE = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def run_step(args: list[str]) -> None:
    print("\n$", " ".join(str(a) for a in args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run preprocessing, IBS, and ERP analyses.")
    parser.add_argument("--skip-preprocess", action="store_true", help="Use existing results/cleaned_eeg.pkl.")
    parser.add_argument("--skip-ibs", action="store_true", help="Do not run IBS analysis.")
    parser.add_argument("--skip-erp", action="store_true", help="Do not run ERP analysis.")
    parser.add_argument("--data-dir", default=str(ROOT), help="Folder containing G01_eeg.mat ... G11_eeg.mat.")
    parser.add_argument("--output-dir", default=str(RESULTS), help="Folder for all output files.")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    cache = Path(args.output_dir) / "cleaned_eeg.pkl"

    if not args.skip_preprocess:
        run_step(
            [
                sys.executable,
                str(PIPELINE / "preprocess.py"),
                "--data_dir",
                args.data_dir,
                "--output_dir",
                args.output_dir,
            ]
        )

    if not args.skip_ibs:
        run_step(
            [
                sys.executable,
                str(PIPELINE / "ibs_analysis.py"),
                "--cache",
                str(cache),
                "--output_dir",
                args.output_dir,
            ]
        )

    if not args.skip_erp:
        run_step(
            [
                sys.executable,
                str(PIPELINE / "erp_analysis.py"),
                "--cache",
                str(cache),
                "--output_dir",
                args.output_dir,
            ]
        )


if __name__ == "__main__":
    main()
