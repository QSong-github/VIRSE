#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# VIRSE — Variational Inference for RNA Structure Ensembles
"""
In-silico benchmark: grid sweep over sequence length (D) and sample size (N).

Compares three methods on synthetic Beta-Bernoulli mixture data:
  - PureEM   : standard Expectation-Maximisation (DREEM-style)
  - VIRSE    : mean-approx variational Bayes
  - VIRSE_VI : standard conjugate VB-EM
  - Gibbs    : blocked Gibbs sampler  [optional, slow]

For each (K, D, N) combination the script computes:
  - ARI   : Adjusted Rand Index   (higher is better)
  - pi_MAE: Mean Absolute Error of cluster proportions  (lower is better)
  - mu_KL : Bernoulli KL divergence of mutation profiles (lower is better)

Results are written as CSV files and bar-chart PDFs into --out/<K>/.

Usage
-----
  python run_benchmark.py [options]

Options
-------
  --K        comma-separated list of K values to benchmark  (default: 3,4,5)
  --D_min    smallest D value           (default: 100)
  --D_max    largest  D value           (default: 1000)
  --D_step   step size for D            (default: 100)
  --N        comma-separated N values   (default: 100,200,300,500,800)
  --seed     random seed                (default: 314)
  --gibbs    include Gibbs sampler (slow — use for small grids only)
  --gibbs_iter  Gibbs iterations for benchmark mode  (default: 500)
  --out      output directory           (default: benchmark_results/)

Example (fast — no Gibbs)
-------
  python run_benchmark.py --K 3,4 --D_max 500 --out results/

Example (all three methods, small grid)
-------
  python run_benchmark.py --K 3 --D_max 300 --gibbs --out results/
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score

from EM import PureEM
from varisem import VIRSE, VIRSE_VI, Gibbs


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(N, D, K, seed):
    rng = np.random.RandomState(seed)
    pi_true = rng.dirichlet(np.ones(K) * 2.0)
    mu_true = rng.beta(2.0, 5.0, size=(K, D))
    Z_true  = rng.choice(K, size=N, p=pi_true)
    X       = rng.binomial(1, mu_true[Z_true]).astype(float)
    return X, Z_true, pi_true, mu_true


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def kl_bernoulli(p, q, eps=1e-10):
    """Per-element KL(Bernoulli(p) || Bernoulli(q)), summed over positions."""
    p = np.clip(p, eps, 1 - eps)
    q = np.clip(q, eps, 1 - eps)
    return float(np.sum(p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))))


def align_clusters(mu_true, mu_inferred):
    """Hungarian matching: returns permutation index array."""
    cost = np.mean((mu_true[:, None, :] - mu_inferred[None, :, :]) ** 2, axis=2)
    _, col_ind = linear_sum_assignment(cost)
    return col_ind


def evaluate(Z_true, Z_pred, pi_true, pi_inf, mu_true, mu_inf, K):
    perm       = align_clusters(mu_true, mu_inf)
    pi_aligned = pi_inf[perm]
    mu_aligned = mu_inf[perm]

    ari    = adjusted_rand_score(Z_true, Z_pred)
    pi_mae = float(np.mean(np.abs(pi_true - pi_aligned)))
    mu_kl  = float(np.mean([kl_bernoulli(mu_true[k], mu_aligned[k]) for k in range(K)]))
    return ari, pi_mae, mu_kl


# ---------------------------------------------------------------------------
# Per-cell runner
# ---------------------------------------------------------------------------

def run_cell(X, Z_true, pi_true, mu_true, K, seed, use_gibbs, gibbs_iter):
    records = []

    # --- PureEM ---
    em = PureEM(X, K, random_state=seed)
    em.fit()
    z_em = np.argmax(em.resps, axis=1)
    ari, pi_mae, mu_kl = evaluate(Z_true, z_em, pi_true, em.final_pi, mu_true, em.final_mu, K)
    records.append({'method': 'PureEM', 'ARI': ari, 'pi_MAE': pi_mae, 'mu_KL': mu_kl})

    # --- VIRSE (mean-approx VB) ---
    virse = VIRSE(X, K, random_state=seed)
    virse.fit(verbose_every=0)
    z_virse = np.argmax(virse.q_nk, axis=1)
    ari, pi_mae, mu_kl = evaluate(
        Z_true, z_virse, pi_true,
        virse.pi_inferred_mean, mu_true, virse.mu_inferred_mean, K
    )
    records.append({'method': 'VIRSE', 'ARI': ari, 'pi_MAE': pi_mae, 'mu_KL': mu_kl})

    # --- VIRSE_VI (conjugate VB-EM) ---
    vi = VIRSE_VI(X, K, random_state=seed)
    vi.fit(verbose_every=0)
    z_vi = np.argmax(vi.q_nk, axis=1)
    ari, pi_mae, mu_kl = evaluate(
        Z_true, z_vi, pi_true,
        vi.pi_inferred_mean, mu_true, vi.mu_inferred_mean, K
    )
    records.append({'method': 'VIRSE_VI', 'ARI': ari, 'pi_MAE': pi_mae, 'mu_KL': mu_kl})

    # --- Gibbs (optional) ---
    if use_gibbs:
        g = Gibbs(X, K, n_iter=gibbs_iter, burn_in=gibbs_iter // 4,
                  random_state=seed)
        g.fit(verbose_every=0)
        res = g.get_results()
        ari, pi_mae, mu_kl = evaluate(
            Z_true, res['cluster_assignments'], pi_true,
            res['pi_inferred_mean'], mu_true, res['mu_inferred_mean'], K
        )
        records.append({'method': 'Gibbs', 'ARI': ari, 'pi_MAE': pi_mae, 'mu_KL': mu_kl})

    return records


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

COLORS = {'PureEM': 'gray', 'VIRSE': '#6baed6', 'VIRSE_VI': '#74c476', 'Gibbs': '#fd8d3c'}


def _bar_chart(ax, df_summary, metric, methods, title, ylabel):
    x = np.arange(len(df_summary['D']))
    bw = 0.8 / len(methods)
    offsets = np.linspace(-(len(methods) - 1) / 2, (len(methods) - 1) / 2, len(methods)) * bw
    for method, offset in zip(methods, offsets):
        col = metric + '_' + method
        if col not in df_summary.columns:
            continue
        col_min = metric + '_min_' + method
        col_max = metric + '_max_' + method
        yerr_lo = (df_summary[col] - df_summary[col_min]).clip(lower=0)
        yerr_hi = (df_summary[col_max] - df_summary[col]).clip(lower=0)
        ax.bar(x + offset, df_summary[col],
               width=bw, color=COLORS.get(method, 'steelblue'),
               edgecolor='black', alpha=0.85, label=method,
               yerr=[yerr_lo, yerr_hi], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(df_summary['D'], rotation=45, fontsize=9)
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xlabel('Sequence Length (D)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)


def make_plots(df, K, out_dir, methods):
    """Generate ARI, pi_MAE, mu_KL bar charts aggregated over N per D."""
    agg_cols = {m: ['mean', 'min', 'max'] for m in methods}

    rows = []
    for d, grp in df.groupby('D'):
        row = {'D': d}
        for method in methods:
            sub = grp[grp['method'] == method]
            if sub.empty:
                continue
            for metric in ('ARI', 'pi_MAE', 'mu_KL'):
                row[f'{metric}_{method}']     = sub[metric].mean()
                row[f'{metric}_min_{method}'] = sub[metric].min()
                row[f'{metric}_max_{method}'] = sub[metric].max()
        rows.append(row)
    df_s = pd.DataFrame(rows).sort_values('D')

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    _bar_chart(axes[0], df_s, 'ARI',    methods,
               f'ARI vs. D  (K={K})', 'Adjusted Rand Index')
    _bar_chart(axes[1], df_s, 'pi_MAE', methods,
               f'π MAE vs. D  (K={K})', 'MAE of π')
    _bar_chart(axes[2], df_s, 'mu_KL',  methods,
               f'μ KL-divergence vs. D  (K={K})', 'Mean KL (μ)')
    plt.suptitle(f'Benchmark: K={K}', fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(out_dir, f'benchmark_K{K}.pdf')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f'  Plot saved → {path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_benchmark(K_list, D_values, N_values, seed, out_dir, use_gibbs, gibbs_iter):
    os.makedirs(out_dir, exist_ok=True)
    methods = ['PureEM', 'VIRSE', 'VIRSE_VI'] + (['Gibbs'] if use_gibbs else [])
    total = len(K_list) * len(D_values) * len(N_values)
    done  = 0

    for K in K_list:
        k_dir = os.path.join(out_dir, f'K{K}')
        os.makedirs(k_dir, exist_ok=True)
        all_records = []

        for D in D_values:
            for N in N_values:
                done += 1
                print(f'[{done}/{total}] K={K}  D={D}  N={N}', flush=True)
                X, Z_true, pi_true, mu_true = generate_data(N, D, K, seed)
                recs = run_cell(X, Z_true, pi_true, mu_true, K, seed, use_gibbs, gibbs_iter)
                for r in recs:
                    r['K'] = K
                    r['D'] = D
                    r['N'] = N
                all_records.extend(recs)

        df = pd.DataFrame(all_records)
        csv_path = os.path.join(k_dir, f'results_K{K}.csv')
        df.to_csv(csv_path, index=False)
        print(f'  CSV saved → {csv_path}')

        make_plots(df, K, k_dir, methods)

        # Print summary table
        summary = df.groupby('method')[['ARI', 'pi_MAE', 'mu_KL']].mean().round(4)
        print(f'\n  === K={K} summary (mean over all D, N) ===')
        print(summary.to_string())
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='VIRSE in-silico benchmark: D×N grid sweep.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--K',       type=str, default='3,4,5',
                        help='comma-separated K values to sweep')
    parser.add_argument('--D_min',   type=int, default=100)
    parser.add_argument('--D_max',   type=int, default=1000)
    parser.add_argument('--D_step',  type=int, default=100)
    parser.add_argument('--N',       type=str, default='100,200,300,500,800',
                        help='comma-separated N values')
    parser.add_argument('--seed',    type=int, default=314)
    parser.add_argument('--gibbs',   action='store_true',
                        help='include Gibbs sampler (slow)')
    parser.add_argument('--gibbs_iter', type=int, default=500,
                        help='Gibbs iterations in benchmark mode')
    parser.add_argument('--out',     type=str, default='benchmark_results/')
    args = parser.parse_args()

    K_list   = [int(k) for k in args.K.split(',')]
    D_values = list(range(args.D_min, args.D_max + 1, args.D_step))
    N_values = [int(n) for n in args.N.split(',')]

    run_benchmark(K_list, D_values, N_values, args.seed,
                  args.out, args.gibbs, args.gibbs_iter)
