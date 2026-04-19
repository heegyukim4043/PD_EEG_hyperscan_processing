"""
erp_analysis.py
===============
ERP analysis for three-person hyperscanning EEG (prisoner's dilemma).

Requires cleaned_eeg.pkl produced by preprocess.py.

Analysis
--------
(A) Grand-average ERP at Fz / Cz / Pz  (decision & feedback)
(B) Cooperative vs. Defection ERP comparison  (exploratory)
(C) Difference wave  (Coop - Defect) ± SEM
(D) Group-level peak ERP amplitude  [250–500 ms] at Pz

ERP pipeline
------------
  Trials averaged per subject → low-pass 15 Hz (Butterworth, order 4)
  → baseline correction [-200, 0] ms
  Display window: -500 to 750 ms

Epoch info (from .mat, onset at sample 300 = 1000 ms pre-stimulus)
  decision : [-1000, 4000] ms  (1500 samples @ 300 Hz)
  feedback : [-1000, 2000] ms  ( 900 samples @ 300 Hz)

Output
------
results/
  fig_erp_decision.png    Grand avg + Coop vs Def (decision)
  fig_erp_feedback.png    Grand avg + Coop vs Def (feedback)
  fig_erp_diff.png        Difference wave
  fig_erp_group.png       Group-level peak amplitude
  erp_amplitude_summary.csv

Usage
-----
  python erp_analysis.py
  python erp_analysis.py --cache results/cleaned_eeg.pkl --output_dir results
"""

import os
import argparse
import pickle
import warnings

import numpy as np
import scipy.io as sio
import scipy.signal as sig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ── Default paths ──────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
_CACHE      = os.path.join(BASE_DIR, '..', 'results', 'cleaned_eeg.pkl')
_OUTPUT_DIR = os.path.join(BASE_DIR, '..', 'results')

# ── Recording parameters ───────────────────────────────────────────────────────
SRATE     = 300
GROUP_IDS = list(range(1, 12))
N_SUBJ    = 3
N_CH      = 19
N_TRIALS  = 40

CH_NAMES = ['P3','C3','F3','Fz','F4','C4','P4','Cz','Pz',
            'Fp1','Fp2','T3','T5','O1','O2','F7','F8','T6','T4']

# ── ERP parameters ─────────────────────────────────────────────────────────────
TASKS = {
    'decision': {'onset_ms': 1000, 'n_smp': 1500, 'ep': (-1000, 4000)},
    'feedback': {'onset_ms': 1000, 'n_smp':  900, 'ep': (-1000, 2000)},
}
PLOT_CHS  = {'Fz': 3, 'Cz': 7, 'Pz': 8}    # channel name -> index in CH_NAMES
BL_MS     = (-200, 0)                        # baseline window (ms)
PLOT_XLIM = (-500, 750)                      # display window (ms)
MIN_TRIALS = 3                               # min trials/condition to include subject

# ── Colors ─────────────────────────────────────────────────────────────────────
C_COOP = '#2166AC'
C_DEF  = '#D6604D'
C_ALL  = '#555555'

# ── Low-pass filter for ERP ────────────────────────────────────────────────────
LP_SOS = sig.butter(4, 15, btype='low', fs=SRATE, output='sos')


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def ms_to_smp(ms, onset_ms):
    return int((onset_ms + ms) * SRATE / 1000)


def baseline_correct(erp, onset_ms):
    i0 = ms_to_smp(BL_MS[0], onset_ms)
    i1 = ms_to_smp(BL_MS[1], onset_ms)
    return erp - erp[..., i0:i1].mean(axis=-1, keepdims=True)


def get_time_axis(ep, n_smp):
    return np.linspace(ep[0], ep[1], n_smp)


# ══════════════════════════════════════════════════════════════════════════════
# ERP computation
# ══════════════════════════════════════════════════════════════════════════════
def compute_subject_erp(eeg, trials_mask, onset_ms):
    """Average trials, apply 15 Hz lowpass, baseline-correct.

    Parameters
    ----------
    eeg         : ndarray (n_ch, n_times, n_trials)
    trials_mask : ndarray bool (n_trials,)
    onset_ms    : int  pre-stimulus period in ms

    Returns
    -------
    erp : ndarray (n_ch, n_times) or None if no trials
    """
    if trials_mask.sum() == 0:
        return None
    erp = eeg[:, :, trials_mask].mean(axis=2)
    erp = sig.sosfiltfilt(LP_SOS, erp, axis=1)
    return baseline_correct(erp, onset_ms)


def collect_erps(cache, task):
    """Collect grand-average and condition ERPs across all subjects.

    Returns
    -------
    erps_all   : ndarray (n_subj, n_ch, n_times)
    erps_coop  : ndarray (n_subj_coop, n_ch, n_times)
    erps_def   : ndarray (n_subj_def, n_ch, n_times)
    g_coop     : dict  group_id -> list of subject ERP arrays (coop)
    g_def      : dict  group_id -> list of subject ERP arrays (def)
    """
    cfg      = TASKS[task]
    onset_ms = cfg['onset_ms']

    erps_all, erps_coop, erps_def = [], [], []
    g_coop = {g: [] for g in GROUP_IDS}
    g_def  = {g: [] for g in GROUP_IDS}

    for g in GROUP_IDS:
        eeg_g = cache[g][task]          # (57, n_times, n_trials)
        score = cache[g]['score']       # (40, 3)  1=coop

        for s in range(N_SUBJ):
            sl    = slice(s * N_CH, (s + 1) * N_CH)
            eeg_s = eeg_g[sl, :, :N_TRIALS]
            sc_s  = score[:, s]

            erp_all  = compute_subject_erp(eeg_s, np.ones(N_TRIALS, bool), onset_ms)
            erp_coop = compute_subject_erp(eeg_s, sc_s == 1, onset_ms)
            erp_def  = compute_subject_erp(eeg_s, sc_s == 2, onset_ms)

            if erp_all is not None:
                erps_all.append(erp_all)
            if erp_coop is not None and (sc_s == 1).sum() >= MIN_TRIALS:
                erps_coop.append(erp_coop)
                g_coop[g].append(erp_coop)
            if erp_def is not None and (sc_s == 2).sum() >= MIN_TRIALS:
                erps_def.append(erp_def)
                g_def[g].append(erp_def)

    return (np.array(erps_all), np.array(erps_coop), np.array(erps_def),
            g_coop, g_def)


def grand_avg(erps):
    """(n, ch, t) -> mean (ch, t), sem (ch, t)"""
    return erps.mean(axis=0), erps.std(axis=0, ddof=1) / np.sqrt(erps.shape[0])


# ══════════════════════════════════════════════════════════════════════════════
# Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════
def _add_erp(ax, t, mean, sem, color, label):
    ax.fill_between(t, mean - sem, mean + sem, color=color, alpha=0.18)
    ax.plot(t, mean, color=color, linewidth=1.8, label=label)


def _decorate(ax, task):
    comps = {'decision': [('N2', 200), ('P3', 350)],
             'feedback': [('FRN', 250), ('P300', 380)]}
    for name, lat in comps[task]:
        ax.axvline(lat, color='gray', linewidth=0.8, linestyle=':', alpha=0.6)
        ax.text(lat + 8, ax.get_ylim()[1] * 0.92, name,
                fontsize=7, color='gray', alpha=0.8)


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════
def plot_erp_panels(results, output_dir):
    ch_list = list(PLOT_CHS.items())

    for task, res in results.items():
        cfg = TASKS[task]
        t   = res['t']
        fig, axes = plt.subplots(2, len(ch_list), figsize=(5 * len(ch_list), 9),
                                 sharey='row')
        fig.suptitle(
            f'ERP — {task.capitalize()}\n'
            f'(displayed: {PLOT_XLIM[0]} to {PLOT_XLIM[1]} ms; '
            f'full epoch {cfg["ep"][0]} to {cfg["ep"][1]} ms)\n'
            f'Grand avg N = {res["n_all"]} subjects; '
            f'Coop N = {res["n_coop"]}; Defect N = {res["n_def"]}',
            fontsize=11, fontweight='bold')

        for ci, (ch_name, ch_idx) in enumerate(ch_list):
            ax0 = axes[0, ci]
            _add_erp(ax0, t, res['GA_all'][ch_idx], res['sem_all'][ch_idx],
                     C_ALL, 'Grand average')
            ax0.axvline(0, color='black', lw=1.2, ls='--', label='Stimulus onset')
            ax0.axhline(0, color='black', lw=0.5)
            ax0.axvspan(*BL_MS, alpha=0.07, color='green', label='Baseline')
            ax0.set_title(f'(A) Grand Average — {ch_name}', fontsize=10, fontweight='bold')
            ax0.set_xlim(PLOT_XLIM)
            ax0.set_xlabel('Time (ms)', fontsize=9)
            ax0.set_ylabel('Amplitude (µV)', fontsize=9)
            ax0.legend(fontsize=7, loc='upper right')
            ax0.spines[['top', 'right']].set_visible(False)
            try: _decorate(ax0, task)
            except Exception: pass

            ax1 = axes[1, ci]
            if res['n_coop'] > 0:
                _add_erp(ax1, t, res['GA_coop'][ch_idx], res['sem_coop'][ch_idx],
                         C_COOP, f'Cooperative (n={res["n_coop"]})')
            if res['n_def'] > 0:
                _add_erp(ax1, t, res['GA_def'][ch_idx], res['sem_def'][ch_idx],
                         C_DEF, f'Defection (n={res["n_def"]})')
            ax1.axvline(0, color='black', lw=1.2, ls='--')
            ax1.axhline(0, color='black', lw=0.5)
            ax1.axvspan(*BL_MS, alpha=0.07, color='green')
            ax1.set_title(f'(B) Coop vs. Defect — {ch_name}', fontsize=10, fontweight='bold')
            ax1.set_xlim(PLOT_XLIM)
            ax1.set_xlabel('Time (ms)', fontsize=9)
            ax1.set_ylabel('Amplitude (µV)', fontsize=9)
            ax1.legend(fontsize=7, loc='upper right')
            ax1.spines[['top', 'right']].set_visible(False)
            try: _decorate(ax1, task)
            except Exception: pass

        plt.tight_layout()
        path = os.path.join(output_dir, f'fig_erp_{task}.png')
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {os.path.basename(path)}')


def plot_difference_wave(results, output_dir):
    ch_list = list(PLOT_CHS.items())
    fig, axes = plt.subplots(1, len(ch_list), figsize=(5 * len(ch_list), 4.5),
                             sharey=True)
    fig.suptitle('(C) Difference Wave — Cooperative minus Defection\n'
                 'Exploratory. Shading = ± SEM (combined)',
                 fontsize=12, fontweight='bold')
    task_colors = {'decision': '#2166AC', 'feedback': '#8172B2'}

    for ci, (ch_name, ch_idx) in enumerate(ch_list):
        ax = axes[ci]
        for task, res in results.items():
            if res['n_coop'] == 0 or res['n_def'] == 0:
                continue
            diff  = res['GA_coop'][ch_idx] - res['GA_def'][ch_idx]
            sem_d = np.sqrt(res['sem_coop'][ch_idx]**2 + res['sem_def'][ch_idx]**2)
            ax.fill_between(res['t'], diff - sem_d, diff + sem_d,
                            color=task_colors[task], alpha=0.18)
            ax.plot(res['t'], diff, color=task_colors[task],
                    linewidth=1.8, label=task.capitalize())
        ax.axhline(0, color='black', lw=0.8)
        ax.axvline(0, color='black', lw=1.2, ls='--', label='Onset')
        ax.set_xlim(PLOT_XLIM)
        ax.set_xlabel('Time (ms)', fontsize=10)
        ax.set_ylabel('Amplitude diff (µV)', fontsize=10)
        ax.set_title(ch_name, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig_erp_diff.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {os.path.basename(path)}')


def plot_group_amplitude(results, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle('(D) Group-level Peak ERP Amplitude\n'
                 '[250–500 ms] at Pz, Cooperative vs. Defection',
                 fontsize=12, fontweight='bold')
    pz = PLOT_CHS['Pz']
    x  = np.arange(len(GROUP_IDS))
    w  = 0.35
    xlbls = [f'G{g:02d}' for g in GROUP_IDS]

    for ti, (task, res) in enumerate(results.items()):
        ax   = axes[ti]
        t    = res['t']
        win  = (t >= 250) & (t <= 500)
        coop_amp, def_amp = [], []
        for g in GROUP_IDS:
            gc = res['g_coop'][g]
            gd = res['g_def'][g]
            coop_amp.append(np.mean([e[pz, win].mean() for e in gc]) if gc else np.nan)
            def_amp.append(np.mean([e[pz, win].mean() for e in gd]) if gd else np.nan)
        ca = np.array(coop_amp); da = np.array(def_amp)
        vc = ~np.isnan(ca);      vd = ~np.isnan(da)
        ax.bar(x[vc] - w/2, ca[vc], w, color=C_COOP, label='Cooperative',
               edgecolor='white', lw=0.7)
        ax.bar(x[vd] + w/2, da[vd], w, color=C_DEF, label='Defection',
               edgecolor='white', lw=0.7)
        ax.axhline(0, color='black', lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(xlbls, rotation=45, fontsize=8)
        ax.set_ylabel('Mean amplitude [250–500 ms] (µV)', fontsize=9)
        ax.set_title(f'{task.capitalize()} — Pz', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, 'fig_erp_group.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {os.path.basename(path)}')


# ══════════════════════════════════════════════════════════════════════════════
# CSV export
# ══════════════════════════════════════════════════════════════════════════════
def export_csv(results, output_dir):
    import csv
    rows = []
    windows_def = [('FRN_N2_window', 150, 250), ('P300_window', 250, 500)]
    for task, res in results.items():
        t = res['t']
        for ch_name, ch_idx in PLOT_CHS.items():
            for win_name, w0, w1 in windows_def:
                mask = (t >= w0) & (t <= w1)
                ga   = res['GA_all'][ch_idx, mask].mean()
                co   = res['GA_coop'][ch_idx, mask].mean() if res['n_coop'] > 0 else np.nan
                de   = res['GA_def'][ch_idx, mask].mean()  if res['n_def']  > 0 else np.nan
                diff = co - de if not (np.isnan(co) or np.isnan(de)) else np.nan
                rows.append({
                    'task': task, 'channel': ch_name,
                    'window': win_name, 'window_ms': f'{w0}-{w1}',
                    'grand_avg_uV': f'{ga:.4f}',
                    'coop_uV':  f'{co:.4f}' if not np.isnan(co) else 'NA',
                    'def_uV':   f'{de:.4f}' if not np.isnan(de) else 'NA',
                    'coop_minus_def': f'{diff:.4f}' if not np.isnan(diff) else 'NA',
                    'n_coop': res['n_coop'], 'n_def': res['n_def'],
                })
    path = os.path.join(output_dir, 'erp_amplitude_summary.csv')
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    print(f'  Saved: {os.path.basename(path)}')


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description='Hyperscanning ERP analysis')
    p.add_argument('--cache',      default=_CACHE,
                   help='Path to cleaned_eeg.pkl from preprocess.py')
    p.add_argument('--output_dir', default=_OUTPUT_DIR,
                   help='Directory to write figures and CSV')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print('=== ERP Analysis ===')
    print(f'  cache      : {args.cache}')
    print(f'  output_dir : {args.output_dir}')

    print('\nLoading cache...')
    with open(args.cache, 'rb') as f:
        cache = pickle.load(f)

    results = {}
    for task, cfg in TASKS.items():
        print(f'\n[{task.capitalize()}]', flush=True)
        erps_all, erps_coop, erps_def, g_coop, g_def = collect_erps(cache, task)
        n_all, n_coop, n_def = len(erps_all), len(erps_coop), len(erps_def)
        print(f'  Grand avg N={n_all}  Coop N={n_coop}  Def N={n_def}')

        GA_all,  sem_all  = grand_avg(erps_all)
        GA_coop, sem_coop = (grand_avg(erps_coop) if n_coop > 0
                             else (np.zeros_like(GA_all),) * 2)
        GA_def,  sem_def  = (grand_avg(erps_def)  if n_def  > 0
                             else (np.zeros_like(GA_all),) * 2)

        results[task] = dict(
            t=get_time_axis(cfg['ep'], cfg['n_smp']),
            GA_all=GA_all, sem_all=sem_all,
            GA_coop=GA_coop, sem_coop=sem_coop,
            GA_def=GA_def,  sem_def=sem_def,
            n_all=n_all, n_coop=n_coop, n_def=n_def,
            g_coop=g_coop, g_def=g_def,
        )

    print('\nPlotting...')
    plot_erp_panels(results, args.output_dir)
    plot_difference_wave(results, args.output_dir)
    plot_group_amplitude(results, args.output_dir)

    print('\nExporting CSV...')
    export_csv(results, args.output_dir)

    print('\nDone.')


if __name__ == '__main__':
    main()
