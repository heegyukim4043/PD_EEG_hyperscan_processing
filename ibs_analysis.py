"""
ibs_analysis.py
===============
Inter-Brain Synchrony (IBS) analysis for three-person hyperscanning EEG.

Requires cleaned_eeg.pkl produced by preprocess.py.

Analysis
--------
- Metrics  : PLV, Coherence (ccorr) via HyPyP
- Windows  : 0–1000 ms, 0–2000 ms post-stimulus onset
- Tasks    : decision, feedback
- Pairs    : all 3 dyads within each trio (S1-S2, S1-S3, S2-S3)
- Condition: Cooperative (both chose cooperate) vs. Other
- Resting  : non-overlapping windows matched to task window length (baseline)
- Stats    : cluster-permutation test (HyPyP, 2000 permutations, two-tailed)

Frequency bands
---------------
  delta  1–3 Hz
  theta  4–7 Hz
  alpha  8–12 Hz
  beta  14–25 Hz
  gamma 30–45 Hz

Output
------
results/
  ibs_data.pkl          IBS arrays per task / window / metric / condition
  cluster_stats.pkl     Cluster-permutation results (min_p, n_sig, F_obs)
  stats_ibs_by_band.csv Per-band summary (mean ± SEM, Cohen's d, cluster p)
  fig_ibs_plv.png       Bar figure — PLV
  fig_ibs_coh.png       Bar figure — Coherence

Usage
-----
  python ibs_analysis.py
  python ibs_analysis.py --cache results/cleaned_eeg.pkl --output_dir results
"""

import os
import argparse
import pickle
import warnings

import numpy as np
import scipy.signal as sig
import scipy.sparse as sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mne
import hypyp.analyses as hana
import hypyp.stats as hstats

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ── Default paths ──────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
_CACHE      = os.path.abspath(os.path.join(BASE_DIR, '..', 'results', 'cleaned_eeg.pkl'))
_OUTPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'results'))

# ── Recording parameters ───────────────────────────────────────────────────────
SRATE     = 300
GROUP_IDS = list(range(1, 12))
N_SUBJ    = 3
N_CH      = 19
N_TRIALS  = 40

CH_NAMES = ['P3','C3','F3','Fz','F4','C4','P4','Cz','Pz',
            'Fp1','Fp2','T3','T5','O1','O2','F7','F8','T6','T4']

MONTAGE = mne.channels.make_standard_montage('standard_1020')

# ── Frequency bands ────────────────────────────────────────────────────────────
BANDS = {
    'delta': ( 1,  3),
    'theta': ( 4,  7),
    'alpha': ( 8, 12),
    'beta' : (14, 25),
    'gamma': (30, 45),
}
BAND_NAMES = list(BANDS.keys())
FREQS_MEAN = [2.0, 5.5, 10.0, 19.5, 37.5]   # representative frequency per band
N_BANDS    = len(BANDS)
N_FEAT     = N_BANDS * N_CH   # 95

# ── Subject pairs within each trio ────────────────────────────────────────────
PAIRS       = [(0, 1), (0, 2), (1, 2)]
PAIR_LABELS = ['S1-S2', 'S1-S3', 'S2-S3']

# ── Analysis windows (ms relative to stimulus onset at 0 ms) ──────────────────
WIN_ONSET_SMP = int(1000 * SRATE / 1000)   # pre-stimulus = 1000 ms = 300 samples
WINDOWS    = [(0, 1000), (0, 2000)]
WIN_LABELS = ['0-1000 ms', '0-2000 ms']

# ── Resting state ──────────────────────────────────────────────────────────────
REST_N_RUNS = 3

# ── Statistics ─────────────────────────────────────────────────────────────────
N_PERMS = 2000

# ── Figure colors ──────────────────────────────────────────────────────────────
COND_COLORS = {'coop': '#2166AC', 'other': '#D6604D'}


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════
def ms_to_smp(ms):
    return int(ms * SRATE / 1000)


def _sig_label(p):
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return 'n.s.'


def _per_band_means(arr):
    """arr: (n, 95) → dict  band → (mean, sem)"""
    out = {}
    for bi, band in enumerate(BAND_NAMES):
        chunk = arr[:, bi * N_CH:(bi + 1) * N_CH].mean(axis=1)
        out[band] = (chunk.mean(), chunk.std(ddof=1) / np.sqrt(len(chunk)))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# HyPyP adjacency matrix (memoised per window length)
# ══════════════════════════════════════════════════════════════════════════════
_adj_cache = {}

def get_adjacency(win_len_smp):
    if win_len_smp in _adj_cache:
        return _adj_cache[win_len_smp]
    info = mne.create_info(CH_NAMES, sfreq=SRATE, ch_types='eeg')
    info.set_montage(MONTAGE, on_missing='ignore')
    ep  = mne.EpochsArray(np.zeros((5, N_CH, win_len_smp)), info, verbose=False)
    ch_con = hstats.con_matrix(ep, freqs_mean=FREQS_MEAN, draw=False)
    meta   = hstats.metaconn_matrix_2brains(
        [(i, i) for i in range(N_CH)], ch_con.ch_con,
        freqs_mean=FREQS_MEAN, plot=False)
    adj = sparse.csr_matrix(meta.metaconn_freq)
    _adj_cache[win_len_smp] = adj
    return adj


# ══════════════════════════════════════════════════════════════════════════════
# IBS computation
# ══════════════════════════════════════════════════════════════════════════════
def _band_analytic(eeg_win, flo, fhi):
    """Bandpass + Hilbert for task epochs.

    Parameters
    ----------
    eeg_win : ndarray (n_ch, n_times, n_trials)

    Returns
    -------
    ndarray (n_trials, n_ch, 1, n_times)  complex
    """
    sos  = sig.butter(4, [flo, fhi], btype='bandpass', fs=SRATE, output='sos')
    filt = sig.sosfiltfilt(sos, eeg_win, axis=1)
    anal = sig.hilbert(filt, axis=1)
    return anal.transpose(2, 0, 1)[:, :, np.newaxis, :]


def compute_task_ibs(eeg_a, eeg_b, win_start_smp, win_end_smp):
    """PLV and Coherence (ccorr) between two subjects across all task trials.

    Parameters
    ----------
    eeg_a, eeg_b     : ndarray (n_ch, n_times, n_trials)
    win_start_smp    : int  start sample index within epoch
    win_end_smp      : int  end sample index within epoch

    Returns
    -------
    plv_mat, coh_mat : ndarray (n_trials, N_FEAT=95)
    """
    seg_a = eeg_a[:, win_start_smp:win_end_smp, :]
    seg_b = eeg_b[:, win_start_smp:win_end_smp, :]
    plv_bands, coh_bands = [], []
    for (flo, fhi) in BANDS.values():
        cs    = np.stack([_band_analytic(seg_a, flo, fhi),
                          _band_analytic(seg_b, flo, fhi)], axis=0)
        plv_f = hana.compute_sync(cs, mode='plv',   epochs_average=False)
        coh_f = hana.compute_sync(cs, mode='ccorr', epochs_average=False)
        plv_ibs = plv_f[0, :, 0:N_CH, N_CH:2*N_CH]
        coh_ibs = coh_f[0, :, 0:N_CH, N_CH:2*N_CH]
        plv_bands.append(np.array([np.diag(plv_ibs[t]) for t in range(N_TRIALS)]))
        coh_bands.append(np.array([np.diag(np.real(coh_ibs[t])) for t in range(N_TRIALS)]))
    return (np.concatenate(plv_bands, axis=1),
            np.concatenate(coh_bands, axis=1))


def compute_resting_ibs(rest_a, rest_b, win_len_smp):
    """PLV and Coherence for resting-state non-overlapping windows.

    Parameters
    ----------
    rest_a, rest_b : ndarray (n_ch, n_times)  µV, 1–45 Hz filtered
    win_len_smp    : int  window length in samples

    Returns
    -------
    plv_mat, coh_mat : ndarray (n_windows, N_FEAT=95)
    """
    n_windows = rest_a.shape[1] // win_len_smp
    plv_bands, coh_bands = [], []
    for (flo, fhi) in BANDS.values():
        sos    = sig.butter(4, [flo, fhi], btype='bandpass', fs=SRATE, output='sos')
        anal_a = sig.hilbert(sig.sosfiltfilt(sos, rest_a, axis=1), axis=1)
        anal_b = sig.hilbert(sig.sosfiltfilt(sos, rest_b, axis=1), axis=1)
        plv_wins, coh_wins = [], []
        for w in range(n_windows):
            sl  = slice(w * win_len_smp, (w + 1) * win_len_smp)
            sa, sb = anal_a[:, sl], anal_b[:, sl]
            phi = np.angle(sa) - np.angle(sb)
            plv = np.abs(np.mean(np.exp(1j * phi), axis=1))
            cross = sa * np.conj(sb)
            coh   = (np.abs(np.mean(cross, axis=1)) /
                     (np.sqrt(np.mean(np.abs(sa)**2, axis=1) *
                              np.mean(np.abs(sb)**2, axis=1)) + 1e-12))
            plv_wins.append(plv)
            coh_wins.append(coh)
        plv_bands.append(np.array(plv_wins))
        coh_bands.append(np.array(coh_wins))
    return (np.concatenate(plv_bands, axis=1),
            np.concatenate(coh_bands, axis=1))


# ══════════════════════════════════════════════════════════════════════════════
# Aggregate IBS across all groups
# ══════════════════════════════════════════════════════════════════════════════
def aggregate_ibs(cache):
    """Compute IBS for all tasks / windows / pairs and resting state.

    Returns
    -------
    results : dict
        results[task][(start_ms, end_ms)][metric]['coop']  → ndarray (n, 95)
        results[task][(start_ms, end_ms)][metric]['other'] → ndarray (n, 95)
        results['resting'][dur_ms][metric]['all']           → ndarray (n, 95)
    """
    tasks = {'decision': 'decision', 'feedback': 'feedback'}

    results = {}
    for task in tasks:
        results[task] = {}
        for win in WINDOWS:
            results[task][win] = {m: {'coop': [], 'other': []}
                                  for m in ('plv', 'coh')}

    rest_durations = sorted(set(end - start for start, end in WINDOWS))
    results['resting'] = {d: {m: {'all': []} for m in ('plv', 'coh')}
                          for d in rest_durations}

    for g in GROUP_IDS:
        score = cache[g]['score']   # (40, 3)  1 = coop

        for task in tasks:
            cleaned = cache[g][task]   # (57, n_times, 40)
            for (start_ms, end_ms) in WINDOWS:
                w_start = WIN_ONSET_SMP + ms_to_smp(start_ms)
                w_end   = WIN_ONSET_SMP + ms_to_smp(end_ms)
                for sa, sb in PAIRS:
                    mask = (score[:, sa] == 1) & (score[:, sb] == 1)
                    ea   = cleaned[sa * N_CH:(sa + 1) * N_CH]
                    eb   = cleaned[sb * N_CH:(sb + 1) * N_CH]
                    plv_mat, coh_mat = compute_task_ibs(ea, eb, w_start, w_end)
                    key = (start_ms, end_ms)
                    results[task][key]['plv']['coop'].append(plv_mat[mask])
                    results[task][key]['plv']['other'].append(plv_mat[~mask])
                    results[task][key]['coh']['coop'].append(coh_mat[mask])
                    results[task][key]['coh']['other'].append(coh_mat[~mask])

        for run_idx in range(REST_N_RUNS):
            rest = cache[g]['resting'][run_idx]   # (57, 18000)
            for sa, sb in PAIRS:
                ra = rest[sa * N_CH:(sa + 1) * N_CH]
                rb = rest[sb * N_CH:(sb + 1) * N_CH]
                for dur_ms in rest_durations:
                    plv_r, coh_r = compute_resting_ibs(ra, rb, ms_to_smp(dur_ms))
                    results['resting'][dur_ms]['plv']['all'].append(plv_r)
                    results['resting'][dur_ms]['coh']['all'].append(coh_r)

        print(f'  G{g:02d} done.', flush=True)

    # Concatenate
    for task in tasks:
        for win in WINDOWS:
            for metric in ('plv', 'coh'):
                for cond in ('coop', 'other'):
                    results[task][win][metric][cond] = np.concatenate(
                        results[task][win][metric][cond], axis=0)
    for dur_ms in rest_durations:
        for metric in ('plv', 'coh'):
            results['resting'][dur_ms][metric]['all'] = np.concatenate(
                results['resting'][dur_ms][metric]['all'], axis=0)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Cluster-permutation test
# ══════════════════════════════════════════════════════════════════════════════
def run_cluster_test(coop, other, win_len_smp):
    adj    = get_adjacency(win_len_smp)
    result = hstats.statscondCluster(
        data=[coop, other],
        freqs_mean=FREQS_MEAN,
        ch_con_freq=adj,
        tail=0,
        n_permutations=N_PERMS,
        alpha=0.05,
    )
    pvs   = result.cluster_p_values
    min_p = min(pvs) if len(pvs) else 1.0
    n_sig = sum(p < 0.05 for p in pvs)
    return min_p, n_sig, result


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════
def plot_ibs_bars(results, cluster_stats, output_dir):
    metric_labels = {'plv': 'PLV', 'coh': 'Coherence (ccorr)'}
    tasks = [t for t in results if t != 'resting']

    for metric in ('plv', 'coh'):
        fig, axes = plt.subplots(len(tasks), len(WINDOWS),
                                 figsize=(5 * len(WINDOWS), 4.5 * len(tasks)),
                                 sharey='row')
        fig.suptitle(
            f'Inter-Brain Synchrony — {metric_labels[metric]}\n'
            f'Cooperative vs. Other  (mean ± SEM, cluster permutation n={N_PERMS})',
            fontsize=12, fontweight='bold')

        x = np.arange(N_BANDS)
        w = 0.35
        band_caps = [b.capitalize() for b in BAND_NAMES]

        for ri, task in enumerate(tasks):
            for ci, (win, win_lbl) in enumerate(zip(WINDOWS, WIN_LABELS)):
                ax       = axes[ri, ci]
                coop_bm  = _per_band_means(results[task][win][metric]['coop'])
                other_bm = _per_band_means(results[task][win][metric]['other'])

                for bi, band in enumerate(BAND_NAMES):
                    c_m, c_s = coop_bm[band]
                    o_m, o_s = other_bm[band]
                    ax.bar(x[bi] - w/2, c_m, w, color=COND_COLORS['coop'],
                           yerr=c_s, capsize=3, error_kw=dict(elinewidth=1),
                           label='Cooperative' if bi == 0 else '')
                    ax.bar(x[bi] + w/2, o_m, w, color=COND_COLORS['other'],
                           yerr=o_s, capsize=3, error_kw=dict(elinewidth=1),
                           label='Other' if bi == 0 else '')

                min_p, n_sig = cluster_stats[task][metric][win]['min_p'], \
                               cluster_stats[task][metric][win]['n_sig']
                slbl = _sig_label(min_p)
                if min_p < 0.001:
                    p_str = f'cluster p < 0.001  {slbl}'
                elif min_p < 0.01:
                    p_str = f'cluster p < 0.01  {slbl}'
                elif min_p < 0.05:
                    p_str = f'cluster p = {min_p:.3f}  {slbl}'
                else:
                    p_str = f'cluster p = {min_p:.3f}  n.s.'
                color_sig = 'red' if min_p < 0.05 else 'dimgray'
                ax.text(0.98, 0.97, p_str, transform=ax.transAxes,
                        ha='right', va='top', fontsize=9, color=color_sig,
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white', edgecolor=color_sig, alpha=0.85))

                ax.set_xticks(x)
                ax.set_xticklabels(band_caps, fontsize=10)
                ax.set_title(f'{task.capitalize()}  [{win_lbl}]', fontsize=11)
                ax.set_ylabel(metric_labels[metric], fontsize=10)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                if ri == 0 and ci == 0:
                    ax.legend(fontsize=8, loc='upper left')

        plt.tight_layout()
        path = os.path.join(output_dir, f'fig_ibs_{metric}.png')
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {os.path.basename(path)}')


def plot_fstat(cluster_stats, output_dir):
    for task in ('decision', 'feedback'):
        for metric in ('plv', 'coh'):
            sig_wins = [(win, cluster_stats[task][metric][win])
                        for win in WINDOWS
                        if cluster_stats[task][metric][win]['n_sig'] > 0]
            if not sig_wins:
                continue
            n = len(sig_wins)
            fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.5))
            if n == 1:
                axes = [axes]
            fig.suptitle(f'Cluster F-statistic — {task.capitalize()} {metric.upper()}',
                         fontsize=12, fontweight='bold')
            for ax, (win, info) in zip(axes, sig_wins):
                F   = info['F_obs'].reshape(N_BANDS, N_CH)
                lim = np.abs(F).max()
                im  = ax.imshow(F, aspect='auto', cmap='RdBu_r',
                                vmin=-lim, vmax=lim)
                ax.set_xticks(range(N_CH))
                ax.set_xticklabels(CH_NAMES, rotation=90, fontsize=7)
                ax.set_yticks(range(N_BANDS))
                ax.set_yticklabels([b.capitalize() for b in BAND_NAMES], fontsize=9)
                min_p = info['min_p']
                p_str = '< 0.001' if min_p < 0.001 else f'= {min_p:.3f}'
                ax.set_title(f'[{win[0]}-{win[1]} ms]  p {p_str}', fontsize=10)
                plt.colorbar(im, ax=ax, fraction=0.046, label='F-statistic')
            plt.tight_layout()
            path = os.path.join(output_dir, f'fig_fstat_{task}_{metric}.png')
            plt.savefig(path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f'  Saved: {os.path.basename(path)}')


# ══════════════════════════════════════════════════════════════════════════════
# CSV export
# ══════════════════════════════════════════════════════════════════════════════
def export_csv(results, cluster_stats, output_dir):
    import csv

    rows = []
    for task in ('decision', 'feedback'):
        for win in WINDOWS:
            win_lbl  = f'{win[0]}-{win[1]}ms'
            for metric in ('plv', 'coh'):
                info     = cluster_stats[task][metric][win]
                coop_bm  = _per_band_means(results[task][win][metric]['coop'])
                other_bm = _per_band_means(results[task][win][metric]['other'])
                for bi, band in enumerate(BAND_NAMES):
                    c_m, c_s = coop_bm[band]
                    o_m, o_s = other_bm[band]
                    diff = c_m - o_m
                    nc = results[task][win][metric]['coop'].shape[0]
                    no = results[task][win][metric]['other'].shape[0]
                    ca = results[task][win][metric]['coop'][:, bi*N_CH:(bi+1)*N_CH].mean(axis=1)
                    oa = results[task][win][metric]['other'][:, bi*N_CH:(bi+1)*N_CH].mean(axis=1)
                    pool = np.sqrt(((nc-1)*ca.std(ddof=1)**2 + (no-1)*oa.std(ddof=1)**2) / (nc+no-2))
                    rows.append({
                        'task': task, 'window': win_lbl, 'metric': metric, 'band': band,
                        'coop_mean': f'{c_m:.6f}', 'coop_sem': f'{c_s:.6f}',
                        'other_mean': f'{o_m:.6f}', 'other_sem': f'{o_s:.6f}',
                        'diff': f'{diff:.6f}',
                        'cohens_d': f'{diff / (pool + 1e-12):.4f}',
                        'cluster_p': f'{info["min_p"]:.4f}',
                        'n_sig_clusters': info['n_sig'],
                        'sig': _sig_label(info['min_p']),
                    })

    path = os.path.join(output_dir, 'stats_ibs_by_band.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f'  Saved: {os.path.basename(path)}')


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description='Hyperscanning IBS analysis')
    p.add_argument('--cache',      default=_CACHE,
                   help='Path to cleaned_eeg.pkl from preprocess.py')
    p.add_argument('--output_dir', default=_OUTPUT_DIR,
                   help='Directory to write figures and CSVs')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print('=== IBS Analysis ===')
    print(f'  cache      : {args.cache}')
    print(f'  output_dir : {args.output_dir}')

    print('\n[1/4] Loading preprocessed cache...')
    with open(args.cache, 'rb') as f:
        cache = pickle.load(f)

    print('\n[2/4] Computing IBS (PLV + Coherence)...')
    results = aggregate_ibs(cache)
    with open(os.path.join(args.output_dir, 'ibs_data.pkl'), 'wb') as f:
        pickle.dump(results, f)

    print('\n[3/4] Cluster-permutation tests...')
    cluster_stats = {}
    for task in ('decision', 'feedback'):
        cluster_stats[task] = {'plv': {}, 'coh': {}}
        for (start_ms, end_ms) in WINDOWS:
            win     = (start_ms, end_ms)
            win_smp = ms_to_smp(end_ms - start_ms)
            for metric in ('plv', 'coh'):
                coop  = results[task][win][metric]['coop']
                other = results[task][win][metric]['other']
                print(f'  {task} [{start_ms}-{end_ms}ms] {metric.upper()}'
                      f'  coop n={coop.shape[0]}, other n={other.shape[0]}',
                      flush=True)
                min_p, n_sig, result = run_cluster_test(coop, other, win_smp)
                print(f'    min_p = {min_p:.4f}  n_sig = {n_sig}  {_sig_label(min_p)}')
                cluster_stats[task][metric][win] = {
                    'min_p': min_p, 'n_sig': n_sig,
                    'F_obs': result.F_obs_plot,
                }

    with open(os.path.join(args.output_dir, 'cluster_stats.pkl'), 'wb') as f:
        pickle.dump(cluster_stats, f)

    print('\n[4/4] Figures and CSV...')
    plot_ibs_bars(results, cluster_stats, args.output_dir)
    plot_fstat(cluster_stats, args.output_dir)
    export_csv(results, cluster_stats, args.output_dir)

    print('\nDone.')


if __name__ == '__main__':
    main()
