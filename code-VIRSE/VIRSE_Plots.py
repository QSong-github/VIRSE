#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# VIRSE — Variational Inference for RNA Structure Ensembles
"""
Output plots and result files for each clustering run.
"""
import os
import plotly
import plotly.graph_objs as go
from plotly import tools
import datetime


def Run_Plots(sample_name, X, K, log_like_list, final_mu, final_obs_pi,
              final_real_pi, resps, BIC, outplots_dir, run):
    """Write result files and HTML plots for one clustering run."""
    run_dir = os.path.join(outplots_dir, f'K_{K}', f'run_{run}')
    os.makedirs(run_dir, exist_ok=True)

    indices = X.indices.split(',')
    start, end = int(indices[0]), int(indices[1])
    seq = X.seq

    # File 1 - List of log likelihoods
    with open(os.path.join(run_dir, 'Log_Likelihoods.txt'), 'w') as f:
        f.write('\n'.join(str(round(x, 2)) for x in log_like_list) + '\n')

    # File 2 - Largest log likelihood
    with open(os.path.join(run_dir, 'Largest_LogLike.txt'), 'w') as f:
        f.write(str(round(log_like_list[-1], 2)) + '\n')

    # File 3 - BIC
    with open(os.path.join(run_dir, 'BIC.txt'), 'w') as f:
        f.write(str(round(BIC, 2)) + '\n')

    # File 4 - Cluster mus
    with open(os.path.join(run_dir, 'Clusters_Mu.txt'), 'w') as f:
        f.write('@ref\t' + X.ref_file + ';' + X.ref + '\t' +
                seq[start - 1:end] + '\n')
        f.write('@coordinates:length\t' + str(start) + ',' +
                str(end) + ':' + str(end - start + 1) + '\n')
        f.write('Position' + ''.join(f'\tCluster_{i+1}' for i in range(len(final_mu))) + '\n')
        for i in range(start, end + 1):
            row = str(i) + ''.join(f'\t{round(final_mu[j][i-start], 5)}' for j in range(len(final_mu)))
            f.write(row + '\n')

    # File 5 - Responsibilities
    with open(os.path.join(run_dir, 'Responsibilities.txt'), 'w') as f:
        header = 'Number\t' + '\t'.join(f'Cluster_{k+1}' for k in range(K)) + '\tN\tBit_vector\n'
        f.write(header)
        for idx, bit_vect in enumerate(X.n_occur, start=1):
            row = str(idx) + '\t'
            row += '\t'.join(str(round(resps[idx-1][k], 3)) for k in range(K))
            row += f'\t{X.n_occur[bit_vect]}\t{"" .join(bit_vect)}\n\n'
            f.write(row)

    # File 6 - Cluster proportions
    with open(os.path.join(run_dir, 'Proportions.txt'), 'w') as f:
        f.write('Cluster, Obs Pi, Real pi\n')
        for k in range(K):
            f.write(f'{k+1},{round(final_obs_pi[k], 2)},{round(final_real_pi[k], 2)}\n')

    # Plot 1 - log likelihood vs iteration number
    loglike_trace = go.Scatter(
        x=[(i+1) for i in range(len(log_like_list))],
        y=log_like_list,
        mode='lines'
    )
    loglike_layout = dict(xaxis=dict(title='Iteration'),
                          yaxis=dict(title='Log likelihood'))
    loglike_data = [loglike_trace]
    loglike_fig = dict(data=loglike_data, layout=loglike_layout)
    plotly.offline.plot(loglike_fig, filename=run_dir +
                        'LogLikes_Iterations.html',
                        auto_open=False)

    # Plot 2 - DMS mod rate for each base in each cluster
    DMSModRate_cluster_data = []
    xaxis_coords = [i for i in range(start, end+1)]
    for k in range(K):
        obs_prob = round(final_obs_pi[k], 2)
        real_prob = round(final_real_pi[k], 2)
        c_name = 'Cluster ' + str(k + 1) + ', obs p=' + str(obs_prob) + \
                 ', real p=' + str(real_prob)
        trace = go.Scatter(
            x=xaxis_coords,
            y=final_mu[k],
            name=c_name,
            mode='lines+markers'
        )
        DMSModRate_cluster_data.append(trace)
    DMSModRate_cluster_layout = dict(xaxis=dict(title='Position (BP)'),
                                     yaxis=dict(title='DMS mod rate'))
    DMSModRate_cluster_fig = dict(data=DMSModRate_cluster_data,
                                  layout=DMSModRate_cluster_layout)
    plotly.offline.plot(DMSModRate_cluster_fig, filename=run_dir +
                        'DMSModRate.html',
                        auto_open=False)

    # Plot 3 - Same as Plot 2, but in subplots
    cmap = {'A': 'red', 'T': 'green', 'G': 'orange', 'C': 'blue'}  # Color map
    colors = [cmap[seq[i]] for i in range(len(seq))]
    ref_bases = [seq[i] for i in range(len(seq))]
    titles = ['Cluster ' + str(k+1) for k in range(K)]
    fig3 = tools.make_subplots(rows=K, cols=1, subplot_titles=titles)
    for k in range(K):
        trace = go.Bar(
            x=xaxis_coords,
            y=final_mu[k],
            text=ref_bases,
            marker=dict(color=colors),
            showlegend=False
        )
        fig3.append_trace(trace, k + 1, 1)
    plotly.offline.plot(fig3, filename=run_dir +
                        'DMSModRate_Clusters.html', auto_open=False)


def NumReads_File(sample_name, X, outplots_dir):
    """Write bit-vector read count summary."""
    with open(os.path.join(outplots_dir, 'BitVectors_Filter.txt'), 'w') as f:
        f.write(f'Number of bit vectors used: {X.n_bitvectors}\n')
        f.write(f'Number of unique bit vectors used: {X.n_unique_bitvectors}\n')
        f.write(f'Number of bit vectors discarded: {X.n_discard}\n')


def LogLikes_File(sample_name, K, RUNS, log_likes, BICs,
                  best_run, outplots_dir):
    """Write per-run log-likelihood and BIC table for a given K."""
    path = os.path.join(outplots_dir, f'K_{K}', 'log_likelihoods.txt')
    with open(path, 'w') as f:
        f.write('Run\tLog_likelihood\tBIC_score\n')
        for run in range(1, RUNS + 1):
            tag = f'{run}-best' if run == best_run else str(run)
            f.write(f'{tag}\t{round(log_likes[run-1], 2)}\t{round(BICs[run-1], 2)}\n')


def Log_File(sample_name, NUM_RUNS, MIN_ITS,
             CONV_CUTOFF, INFO_THRESH, SIG_THRESH, INC_TG,
             norm_bases, K, time_taken, outplots_dir):
    """Write run parameters and timing to a log file."""
    now = datetime.datetime.now()
    with open(os.path.join(outplots_dir, 'log.txt'), 'w') as f:
        f.write(f'Sample: {sample_name}\n')
        f.write(f'Number of EM runs: {NUM_RUNS}\n')
        f.write(f'Minimum number of iterations: {MIN_ITS}\n')
        f.write(f'Convergence cutoff: {CONV_CUTOFF}\n')
        f.write(f'Informative bits threshold: {INFO_THRESH}\n')
        f.write(f'Signal threshold: {SIG_THRESH}\n')
        f.write(f'Include Ts and Gs?: {INC_TG}\n')
        f.write(f'Num bases for normalization: {norm_bases}\n')
        f.write(f'Predicted number of clusters: {K}\n')
        f.write(f'Time taken: {time_taken} mins\n')
        f.write(f'Finished at: {now.strftime("%Y-%m-%d %H:%M")}\n')
