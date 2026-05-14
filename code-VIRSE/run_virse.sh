#!/usr/bin/env bash
# VIRSE — master run script
# Edit the parameters below, then execute:  bash run_virse.sh
set -euo pipefail

# ---------------------------------------------------------------------------
# Environment setup — adjust paths for your system
# ---------------------------------------------------------------------------

# RNAstructure (for --struct mode)
export PATH="${PATH}:../data/binaries/RNAstructure/exe"
export DATAPATH="../data/binaries/RNAstructure/data_tables"

# Conda/micromamba environment (edit to match your setup)
# export PATH="/Users/yourname/micromamba/envs/virse/bin:${PATH}"
# export DATAPATH="/Users/yourname/micromamba/envs/virse/share/rnastructure/data_tables"

# ---------------------------------------------------------------------------
# Run parameters — edit these for each experiment
# ---------------------------------------------------------------------------

INPUT_DIR="../data/DREEM_Input/"   # directory with FASTA + (BAM or FASTQ)
OUTPUT_DIR="../results/"           # output directory (created automatically)

SAMPLE="RRE_invitroDMS"            # sample name
REF="NL43rna"                      # reference sequence name in FASTA
START=7410                         # region start (1-based)
END=7500                           # region end   (1-based)

# Optional flags:
#   --fastq   start from FASTQ files (runs Bowtie2 mapping first)
#   --single  single-end sequencing (default: paired-end)
#   --struct  predict RNA secondary structure with RNAstructure

python3 Run_VIRSE.py \
    "${INPUT_DIR}" "${OUTPUT_DIR}" \
    "${SAMPLE}" "${REF}" "${START}" "${END}" \
    --fastq

