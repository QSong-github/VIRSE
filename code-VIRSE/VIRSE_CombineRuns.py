#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# VIRSE — Variational Inference for RNA Structure Ensembles
"""
Post-process multi-run results: pick best run by log-likelihood,
optionally fold RNA structure and produce scatter plots.
"""
import os
import shutil
import VIRSE_Plots
import VIRSE_ExpandFold
import VIRSE_ScatterClusters


def Collect_BestBIC(sample_name, K, outfiles_dir):
    """Return the BIC of the best run for a given K."""
    loglikes_path = os.path.join(outfiles_dir, f'K_{K}', 'log_likelihoods.txt')
    with open(loglikes_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts[0].endswith('best'):
                return float(parts[2])


def Post_Process(sample_name, K, RUNS, cur_BIC, norm_bases,
                 struct, input_dir, outfiles_dir):
    """Select best run, write summary, optionally fold and scatter-plot."""
    largest_loglike, BICs, log_likes, best_run = float('-inf'), [], [], 1

    for run in range(1, RUNS + 1):
        run_dir = os.path.join(outfiles_dir, f'K_{K}', f'run_{run}')

        with open(os.path.join(run_dir, 'Largest_LogLike.txt')) as f:
            log_like = float(f.readline())
        with open(os.path.join(run_dir, 'BIC.txt')) as f:
            BIC = float(f.readline())

        log_likes.append(log_like)
        BICs.append(BIC)

        if log_like > largest_loglike:
            largest_loglike = log_like
            best_run = run

        os.remove(os.path.join(run_dir, 'Largest_LogLike.txt'))
        os.remove(os.path.join(run_dir, 'BIC.txt'))

    VIRSE_Plots.LogLikes_File(sample_name, K, RUNS, log_likes, BICs,
                           best_run, outfiles_dir)

    # Rename best-run directory
    orig_dir = os.path.join(outfiles_dir, f'K_{K}', f'run_{best_run}')
    new_dir  = os.path.join(outfiles_dir, f'K_{K}', f'run_{best_run}-best')
    shutil.move(orig_dir, new_dir)

    clustmu_file = os.path.join(new_dir, 'Clusters_Mu.txt')

    # Optional: fold with RNAstructure
    if struct == 'True':
        for num_base in [0, 50, 100, 150, 200]:
            VIRSE_ExpandFold.ConstraintFoldDraw(
                input_dir, clustmu_file, num_base, num_base, norm_bases)

    # Optional: scatter plot of reactivities
    if K > 1:
        VIRSE_ScatterClusters.Scatter_Clusters(input_dir, clustmu_file)
