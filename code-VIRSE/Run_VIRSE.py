#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# VIRSE — Variational Inference for RNA Structure Ensembles
"""
VIRSE pipeline entry point.

Steps
-----
  1. (optional) Mapping    — align FASTQ reads with Bowtie2 via Mapping.py
  2. Bit-vector creation   — BitVector.py
  3. VIRSE clustering      — VIRSE_Clustering.py (VIRSE VI, BIC-based K selection)

Usage
-----
  python Run_VIRSE.py <input_dir> <output_dir> <sample> <ref> <START> <END>
                      [--fastq] [--single] [--struct]

Or simply edit and run the `run_virse.sh` bash script in this directory.
"""
import os
import argparse
import time


# ---------------------------------------------------------------------------
# Pipeline configuration — edit defaults here as needed
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    picard_path    = './picard.jar',   # path to picard.jar
    CPUS           = 12,               # threads for alignment + clustering
    L              = 12,               # Bowtie2 seed length
    MAX_FRAG_LEN   = 1000,             # max paired-end fragment length
    qscore_file    = './phred_ascii.txt',
    SUR_BASES      = 10,               # bases flanking deletions
    QSCORE_CUTOFF  = 20,               # min base quality
    MIN_ITS        = 300,              # min VIRSE iterations per run
    INFO_THRESH    = 0.0001,           # informative-bits threshold
    CONV_CUTOFF    = 0.5,              # ELBO convergence threshold
    NUM_RUNS       = 5,                # independent runs per K
    MAX_K          = 3,                # max number of clusters
    SIG_THRESH     = 0.05,             # signal vs noise threshold
    NORM_PERC_BASES = 10,              # % bases used for normalisation
    inc_TG         = False,            # include T/G positions?
)


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def run_pipeline(input_dir, output_dir, sample_name, ref_name,
                 START, END, fastq, paired, struct, cfg):
    """Run the full VIRSE pipeline (mapping → bit-vector → clustering)."""
    start_time = time.time()

    os.makedirs(output_dir, exist_ok=True)

    map_cmd = (
        f'python3 Mapping.py {sample_name} {ref_name} {paired} '
        f'{cfg["CPUS"]} {cfg["L"]} {cfg["MAX_FRAG_LEN"]} '
        f'{input_dir} {output_dir} {cfg["picard_path"]}'
    )
    bv_cmd = (
        f'python3 BitVector.py {sample_name} {ref_name} {START} {END} '
        f'{cfg["SUR_BASES"]} {cfg["qscore_file"]} {cfg["QSCORE_CUTOFF"]} '
        f'{input_dir} {output_dir} {paired} {cfg["picard_path"]} {fastq}'
    )
    cluster_cmd = (
        f'python3 VIRSE_Clustering.py {sample_name} {ref_name} {START} {END} '
        f'{cfg["MIN_ITS"]} {cfg["INFO_THRESH"]} {cfg["CONV_CUTOFF"]} '
        f'{cfg["NUM_RUNS"]} {cfg["MAX_K"]} {cfg["CPUS"]} '
        f'{cfg["NORM_PERC_BASES"]} {cfg["inc_TG"]} {cfg["SIG_THRESH"]} '
        f'{struct} {input_dir} {output_dir}'
    )

    if fastq:
        print('[Step 1] Mapping FASTQ reads...')
        os.system(map_cmd)

    print('[Step 2] Building bit vectors...')
    os.system(bv_cmd)

    print('[Step 3] Running VIRSE clustering...')
    os.system(cluster_cmd)

    elapsed = round((time.time() - start_time) / 60, 2)
    print(f'Total time: {elapsed} mins')


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='VIRSE pipeline: alignment → bit-vectors → VI clustering')
    parser.add_argument('input_dir',   help='Directory containing input files')
    parser.add_argument('output_dir',  help='Directory for output files')
    parser.add_argument('sample_name', help='Sample name')
    parser.add_argument('ref_name',    help='Reference genome name')
    parser.add_argument('START', type=int, help='Region start (1-based)')
    parser.add_argument('END',   type=int, help='Region end   (1-based)')
    parser.add_argument('--fastq',  action='store_true', help='Start from FASTQ (run mapping)')
    parser.add_argument('--single', action='store_true', help='Single-end sequencing')
    parser.add_argument('--struct', action='store_true', help='Predict secondary structure')
    args = parser.parse_args()

    run_pipeline(
        input_dir   = args.input_dir.rstrip('/') + '/',
        output_dir  = args.output_dir,
        sample_name = args.sample_name,
        ref_name    = args.ref_name,
        START       = args.START,
        END         = args.END,
        fastq       = args.fastq,
        paired      = not args.single,
        struct      = str(args.struct),
        cfg         = DEFAULTS,
    )

