#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# VIRSE — Variational Inference for RNA Structure Ensembles
"""
In-silico simulation benchmark.

Generates synthetic Beta-Bernoulli mixture data and compares three methods:
  - VIRSE     : mean-approx variational Bayes (the VIRSE model)
  - VIRSE_VI  : standard conjugate VB-EM
  - Gibbs     : blocked Gibbs sampler  (skip with --no-gibbs)

Usage
-----
  python run_simulation.py [options]

Options
-------
  --N         number of reads             (default: 500)
  --D         number of sites             (default: 50)
  --K         number of true clusters     (default: 2)
  --seed      random seed                 (default: 42)
  --runs      independent runs per method (default: 5)
  --no-gibbs  skip Gibbs sampler         (much faster)
  --out       output directory            (default: results/)

Example
-------
  python run_simulation.py --N 1000 --D 80 --K 3 --runs 10 --out results/K3/
  python run_simulation.py --N 1000 --D 80 --K 3 --no-gibbs --out results/K3/
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from EM import PureEM
from varisem import VIRSE, VIRSE_VI, Gibbs


# ---------------------------------------------------------------------------
# Data simulation
# ---------------------------------------------------------------------------

def simulate_data(N, D, K, seed=42):
    """
    Generate synthetic reads from a Beta-Bernoulli mixture.

    Returns
    -------
    X        : (N, D) binary array of observed reads
    z_true   : (N,)   true cluster assignments
    pi_true  : (K,)   true cluster proportions
    mu_true  : (K, D) true mutation probabilities
    """
    rng = np.random.RandomState(seed)
    pi_true = rng.dirichlet(np.ones(K) * 5.0)
    mu_true = rng.beta(2.0, 5.0, size=(K, D))
    z_true  = rng.choice(K, size=N, p=pi_true)
    X       = rng.binomial(1, mu_true[z_true]).astype(float)
    return X, z_true, pi_true, mu_true


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def align_to_truth(mu_true, mu_inferred):
    """
    Hungarian-matching permutation so that inferred clusters align to ground truth.

    Returns col_ind such that mu_inferred[col_ind[k]] ≈ mu_true[k].
    """
    cost = np.mean((mu_true[:, None, :] - mu_inferred[None, :, :]) ** 2, axis=2)
    _, col_ind = linear_sum_assignment(cost)
    return col_ind


def cluster_accuracy(z_true, z_pred, K):
    """Permutation-invariant cluster accuracy via Hungarian matching."""
    conf = np.zeros((K, K), dtype=int)
    for t, p in zip(z_true, z_pred):
        conf[t, p] += 1
    row_ind, col_ind = linear_sum_assignment(-conf)
    return conf[row_ind, col_ind].sum() / len(z_true)


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


# ---------------------------------------------------------------------------
# Single-run helper
# ---------------------------------------------------------------------------

def _run_one(method_name, model, z_true, pi_true, mu_true, K):
    """Fit one model, evaluate, and return a result dict."""
    model.fit()

    if hasattr(model, 'get_results'):
        res      = model.get_results()
        pi_inf   = res['pi_inferred_mean']
        mu_inf   = res['mu_inferred_mean']
        z_pred   = res['cluster_assignments']
    else:                                   # PureEM fallback
        pi_inf   = model.final_pi
        mu_inf   = model.final_mu
        z_pred   = np.argmax(model.resps, axis=1)

    perm       = align_to_truth(mu_true, mu_inf)
    mu_aligned = mu_inf[perm]
    pi_aligned = pi_inf[perm]

    acc    = cluster_accuracy(z_true, z_pred, K)
    mu_mae = mae(mu_true, mu_aligned)
    pi_mae = mae(pi_true, pi_aligned)

    return {'method': method_name, 'accuracy': acc, 'mu_mae': mu_mae, 'pi_mae': pi_mae}


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_simulation(N, D, K, seed, runs, out_dir, use_gibbs=True):
    """Run the full simulation benchmark and save results to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    print(f"[VIRSE sim] N={N}  D={D}  K={K}  seed={seed}  runs={runs}")

    X, z_true, pi_true, mu_true = simulate_data(N, D, K, seed)
    print(f"  True pi : {np.round(pi_true, 3)}")

    # Save ground truth
    pd.DataFrame([pi_true], columns=[f'K{k}' for k in range(K)]).to_csv(
        os.path.join(out_dir, 'true_pi.csv'), index=False)
    pd.DataFrame(mu_true).to_csv(
        os.path.join(out_dir, 'true_mu.csv'), index=False)
    pd.DataFrame({'cluster': z_true}).to_csv(
        os.path.join(out_dir, 'true_assignments.csv'), index=False)

    records = []

    for run in range(runs):
        rs = seed + run
        print(f"\n--- Run {run + 1}/{runs}  (seed={rs}) ---")

        # --- PureEM ---
        em  = PureEM(X, K, random_state=rs)
        rec = _run_one('PureEM', em, z_true, pi_true, mu_true, K)
        rec['run'] = run
        records.append(rec)
        print(f"  PureEM   | acc={rec['accuracy']:.3f}  mu_mae={rec['mu_mae']:.4f}")

        # --- VIRSE (mean-approx VB) ---
        virse = VIRSE(X, K, random_state=rs)
        rec   = _run_one('VIRSE', virse, z_true, pi_true, mu_true, K)
        rec['run'] = run
        records.append(rec)
        print(f"  VIRSE    | acc={rec['accuracy']:.3f}  mu_mae={rec['mu_mae']:.4f}")

        # --- VIRSE_VI (conjugate VB-EM) ---
        vi  = VIRSE_VI(X, K, random_state=rs)
        rec = _run_one('VIRSE_VI', vi, z_true, pi_true, mu_true, K)
        rec['run'] = run
        records.append(rec)
        print(f"  VIRSE_VI | acc={rec['accuracy']:.3f}  mu_mae={rec['mu_mae']:.4f}")

        # --- Gibbs ---
        if use_gibbs:
            g   = Gibbs(X, K, random_state=rs)
            rec = _run_one('Gibbs', g, z_true, pi_true, mu_true, K)
            rec['run'] = run
            records.append(rec)
            print(f"  Gibbs    | acc={rec['accuracy']:.3f}  mu_mae={rec['mu_mae']:.4f}")

    df      = pd.DataFrame(records)
    out_csv = os.path.join(out_dir, 'simulation_results.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved → {out_csv}")

    summary = df.groupby('method')[['accuracy', 'mu_mae', 'pi_mae']].mean().round(4)
    print("\n=== Summary (mean over runs) ===")
    print(summary.to_string())
    summary.to_csv(os.path.join(out_dir, 'summary.csv'))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='VIRSE in-silico simulation benchmark.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--N',    type=int,   default=500,       help='number of reads')
    parser.add_argument('--D',    type=int,   default=50,        help='number of sites')
    parser.add_argument('--K',    type=int,   default=2,         help='number of clusters')
    parser.add_argument('--seed', type=int,   default=42,        help='random seed')
    parser.add_argument('--runs',     type=int,  default=5,         help='independent runs per method')
    parser.add_argument('--no-gibbs', action='store_true',           help='skip Gibbs sampler (faster)')
    parser.add_argument('--out',      type=str,  default='results/', help='output directory')
    args = parser.parse_args()

    run_simulation(args.N, args.D, args.K, args.seed, args.runs, args.out,
                   use_gibbs=not args.no_gibbs)
