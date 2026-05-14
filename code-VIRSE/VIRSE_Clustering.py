#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# VIRSE — Variational Inference for RNA Structure Ensembles
"""
Step 2 of the VIRSE pipeline: cluster bit vectors by K using VIRSE VI.
"""
import time
import os
import BitVector_Functions
import VIRSE_Plots
import VIRSE_CombineRuns
import VIRSE_Jobs
import VIRSE_Files


def EM_Clustering():
    """Run VIRSE clustering for each reference region, sweeping K until BIC stops improving."""
    print('Starting VIRSE clustering...')

    for ref in refs_seq:

        start_time = time.time()

        bvfile_basename = f'{sample_name}_{ref}_{START}_{END}'
        outplot_dir = os.path.join(outfiles_dir, bvfile_basename)
        os.makedirs(outplot_dir, exist_ok=True)
        if os.path.exists(os.path.join(outplot_dir, 'log.txt')):
            print('Clustering already done for', bvfile_basename)
            continue

        norm_bases = int((int(END) - int(START)) * NORM_PERC_BASES / 100)

        input_file = os.path.join(output_dir, 'BitVector_Files',
                                  bvfile_basename + '_bitvectors.txt')
        X = VIRSE_Files.Load_BitVectors(input_file, INFO_THRESH, SIG_THRESH,
                                     inc_TG, output_dir)

        K = 1  # Number of clusters
        cur_BIC = float('inf')  # Initialize BIC
        BIC_failed = False  # While test is not passed
        while not BIC_failed and K <= MAX_K:
            print('Working on K =', K)

            RUNS = NUM_RUNS if K != 1 else 1  # Only 1 Run for K=1
            ITS = MIN_ITS if K != 1 else 10  # Only 10 iters for K=1

            for run in range(1, RUNS + 1):
                print('Run number:', run)
                VIRSE_Jobs.Run_EMJob(X, bvfile_basename, ITS, INFO_THRESH,
                                     CONV_CUTOFF, SIG_THRESH,
                                     outplot_dir, K, CPUS, run)

            # Processing of results from the EM runs
            VIRSE_CombineRuns.Post_Process(bvfile_basename, K, RUNS,
                                        cur_BIC, norm_bases, struct,
                                        input_dir, outplot_dir)

            # Check BIC
            latest_BIC = VIRSE_CombineRuns.Collect_BestBIC(bvfile_basename, K,
                                                        outplot_dir)
            if latest_BIC > cur_BIC:  # BIC test has failed
                BIC_failed = True
            cur_BIC = latest_BIC  # Update BIC

            K += 1  # Move on to next K

        end_time = time.time()
        time_taken = round((end_time - start_time) / 60, 2)
        print('Time taken:', time_taken, 'mins')

        # Write params to log file
        VIRSE_Plots.Log_File(bvfile_basename, NUM_RUNS, MIN_ITS,
                          CONV_CUTOFF, INFO_THRESH, SIG_THRESH, inc_TG,
                          norm_bases, K - 2, time_taken, outplot_dir)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='VIRSE Clustering')
    parser.add_argument('sample_name')
    parser.add_argument('ref_name')
    parser.add_argument('START', type=int)
    parser.add_argument('END', type=int)
    parser.add_argument('MIN_ITS', type=int)
    parser.add_argument('INFO_THRESH', type=float)
    parser.add_argument('CONV_CUTOFF', type=float)
    parser.add_argument('NUM_RUNS', type=int)
    parser.add_argument('MAX_K', type=int)
    parser.add_argument('CPUS', type=int)
    parser.add_argument('NORM_PERC_BASES', type=int)
    parser.add_argument('inc_TG')
    parser.add_argument('SIG_THRESH', type=float)
    parser.add_argument('struct')
    parser.add_argument('input_dir')
    parser.add_argument('output_dir')
    args = parser.parse_args()

    sample_name    = args.sample_name
    ref_name       = args.ref_name
    START          = args.START
    END            = args.END
    MIN_ITS        = args.MIN_ITS
    INFO_THRESH    = args.INFO_THRESH
    CONV_CUTOFF    = args.CONV_CUTOFF
    NUM_RUNS       = args.NUM_RUNS
    MAX_K          = args.MAX_K
    CPUS           = args.CPUS
    NORM_PERC_BASES = args.NORM_PERC_BASES
    inc_TG         = args.inc_TG
    SIG_THRESH     = args.SIG_THRESH
    struct         = args.struct
    input_dir      = args.input_dir
    output_dir     = args.output_dir

    refs_seq = BitVector_Functions.Parse_FastaFile(
        os.path.join(input_dir, ref_name + '.fasta'))

    outfiles_dir = os.path.join(output_dir, 'EM_Clustering')
    os.makedirs(outfiles_dir, exist_ok=True)

    EM_Clustering()
