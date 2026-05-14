# VIRSE — Variational Inference for RNA Structure Ensembles

VIRSE is a computational pipeline for detecting RNA structural ensembles from DMS (Dimethyl Sulfate) mutational profiling sequencing data. It clusters mutation bit-vectors into discrete structural states using Bayesian inference, replacing the original EM algorithm of [DREEM](https://github.com/yvesago/DREEM) with principled variational and MCMC approaches.

---

## Table of Contents

1. [Background](#background)
2. [Algorithms](#algorithms)
3. [Repository Structure](#repository-structure)
4. [Installation](#installation)
5. [Usage — Pipeline](#usage--pipeline)
6. [Usage — In-silico Simulation](#usage--in-silico-simulation)
7. [Output Files](#output-files)
8. [Dependencies](#dependencies)

---

## Background

DMS-MaPseq experiments produce per-read mutation bit-vectors: binary vectors where 1 indicates a modified (mutated) base. RNA molecules that adopt different secondary structures have different solvent-exposed nucleotides, and thus different mutation patterns. By clustering these bit-vectors, VIRSE infers the mixture of structural states and their per-nucleotide mutation probabilities.

**Model:**

$$
\pi \sim \text{Dir}(\alpha), \quad \mu_{ki} \sim \text{Beta}(a, b), \quad z_n \sim \text{Cat}(\pi), \quad x_{ni} \mid z_n = k \sim \text{Bernoulli}(\mu_{ki})
$$

---

## Algorithms

VIRSE provides three inference engines in `code-VIRSE/VARISEM.py` and `in_silico_simulation/varisem.py`:

### 1. `VIRSE` — Mean-approximation Variational Bayes (default)

Approximates posterior expectations with their means directly, avoiding digamma evaluations. Numerically stable and fast.

- **E-step:** $q_{nk} \propto \pi_k \cdot \prod_i \mu_{ki}^{x_{ni}} (1 - \mu_{ki})^{1 - x_{ni}}$
- **M-step:** Update $\pi_k$ and $\mu_{ki}$ via Beta/Dirichlet posterior means
- **Convergence:** Surrogate ELBO

### 2. `VARISEM_VI` / `VARISEM` — Standard Conjugate VB-EM

Full variational Bayes with correct ELBO including Dirichlet and Beta entropy terms.

- **E-step:** Uses digamma expectations $\mathbb{E}[\log \pi_k]$ and $\mathbb{E}[\log \mu_{ki}]$
- **M-step:** Conjugate updates of Dirichlet and Beta variational parameters
- **Convergence:** True ELBO with full entropy

### 3. `BetaBernoulliGibbs` — Blocked Gibbs Sampler

A full Bayesian MCMC sampler. Samples $\pi$, $\mu$, and cluster assignments $z$ iteratively from their full conditionals.

- Produces posterior samples, not point estimates
- More accurate but substantially slower than variational methods
- Useful for validation and uncertainty quantification

**Prior defaults** (all methods):

| Parameter | Default | Description |
|---|---|---|
| `alpha_prior` | 10.0 per component | Dirichlet concentration for $\pi$ |
| `a_prior` | 5.0 per feature | Beta shape for $\mu$ (successes) |
| `b_prior` | 10.0 per feature | Beta shape for $\mu$ (failures) |

---

## Repository Structure

```
VIRSE/
├── code-VIRSE/                  # Main DMS sequencing pipeline
│   ├── run_virse.sh             # Master run script — edit and execute this
│   ├── Run_VIRSE.py             # Pipeline entry point
│   ├── VARISEM.py               # Core algorithms: VIRSE, VARISEM_VI, BetaBernoulliGibbs
│   ├── VIRSE_Algorithm.py       # Bridge: BV_Object → VIRSE model
│   ├── VIRSE_Clustering.py      # Clustering loop (sweeps K by BIC)
│   ├── VIRSE_Jobs.py            # Single-run executor with collapse retry
│   ├── VIRSE_Class.py           # Bit-vector data container (BV_Object)
│   ├── VIRSE_Files.py           # Load & filter bit-vector files
│   ├── VIRSE_Functions.py       # Quality filter helpers
│   ├── VIRSE_Plots.py           # Output files and plotly HTML plots
│   ├── VIRSE_CombineRuns.py     # Select best run, optionally fold/scatter
│   ├── VIRSE_ExpandFold.py      # RNA folding via RNAstructure
│   ├── VIRSE_ScatterClusters.py # Scatter plots comparing cluster profiles
│   ├── Mapping.py               # Bowtie2 read alignment
│   ├── BitVector.py             # BAM → bit-vector conversion
│   ├── BitVector_Functions.py   # Bit-vector quality filters
│   └── BitVector_Outputs.py     # Bit-vector output writers
│
└── in_silico_simulation/        # Benchmark on synthetic data
    ├── varisem.py               # VARISEM + BetaBernoulliGibbs classes
    ├── EM.py                    # PureEM (DREEM-style EM) class
    ├── run_simulation.py        # Single experiment: all 3 methods, N runs
    └── run_benchmark.py         # Grid sweep over D × N × K with plots
```

---

## Installation

```bash
# Clone and create environment
git clone https://github.com/your-org/VIRSE.git
cd VIRSE

# With conda / mamba
conda create -n virse python=3.10
conda activate virse

# Install dependencies
pip install numpy scipy scikit-learn pandas plotly matplotlib

# For RNA folding (optional, --struct mode)
# Download RNAstructure from https://rna.urmc.rochester.edu/RNAstructure.html
# and set DATAPATH accordingly
```

---

## Usage — Pipeline

### Quick start

Edit the parameters in `code-VIRSE/run_virse.sh` and run:

```bash
cd code-VIRSE
bash run_virse.sh
```

### Manual invocation

```bash
cd code-VIRSE

# Start from BAM files (bit-vectors already built)
python Run_VIRSE.py data/ results/ my_sample NL43rna 7410 7500

# Start from FASTQ files (runs Bowtie2 mapping first)
python Run_VIRSE.py data/ results/ my_sample NL43rna 7410 7500 --fastq

# With RNA structure prediction
python Run_VIRSE.py data/ results/ my_sample NL43rna 7410 7500 --struct
```

### Key parameters (edit `DEFAULTS` in `Run_VIRSE.py`)

| Parameter | Default | Description |
|---|---|---|
| `MAX_K` | 3 | Maximum number of clusters to try |
| `NUM_RUNS` | 5 | Independent runs per K |
| `MIN_ITS` | 300 | Minimum VIRSE iterations |
| `CONV_CUTOFF` | 0.5 | ELBO convergence threshold |
| `INFO_THRESH` | 0.0001 | Informative-bits filter threshold |
| `CPUS` | 12 | Threads for alignment and clustering |

### Python API

```python
from VARISEM import VIRSE, VARISEM_VI, BetaBernoulliGibbs, run_varisem

# One-liner convenience
result = run_varisem(X, K=2, method='virse')   # or method='vi'

# Full control — mean-approx VB (default pipeline method)
model = VIRSE(X, K=2, alpha_prior=10.0, a_prior=5.0, b_prior=10.0)
model.fit(verbose_every=50)
results = model.get_results()
# keys: pi_k_mean, mu_ki_mean, cluster_assignments, q_nk, elbo_values

# Standard conjugate VB-EM
model = VARISEM_VI(X, K=2)
model.fit()

# Gibbs sampler
sampler = BetaBernoulliGibbs(X, K=2, n_iter=4000, burn_in=1000)
sampler.fit()
results = sampler.get_results()
# keys: pi_inferred_mean, mu_inferred_mean, cluster_assignments,
#       pi_samples, mu_samples, z_samples
```

---

## Usage — In-silico Simulation

### Single experiment

Runs all 3 methods on one synthetic dataset with multiple independent repeats.

```bash
cd in_silico_simulation

# All 3 methods, 5 runs
python run_simulation.py --N 500 --D 50 --K 2 --runs 5 --out results/

# Skip Gibbs sampler (much faster)
python run_simulation.py --N 500 --D 50 --K 2 --runs 5 --no-gibbs --out results/
```

Outputs: `true_pi.csv`, `true_mu.csv`, `true_assignments.csv`,
`simulation_results.csv` (per-run metrics), `summary.csv` (mean over runs).

### Grid benchmark

Sweeps D × N for multiple K values. Produces CSVs and bar-chart PDFs comparing methods.

```bash
# PureEM + VARISEM, K = 3,4,5, D from 100 to 1000
python run_benchmark.py --K 3,4,5 --D_max 1000 --out benchmark/

# All three methods on a small grid
python run_benchmark.py --K 3 --D_max 300 --N 100,200 --gibbs --out benchmark/
```

Outputs per K: `benchmark/K3/results_K3.csv` and `benchmark/K3/benchmark_K3.pdf`
(3-panel bar chart: ARI, π MAE, μ KL-divergence vs. sequence length D).

**Metrics:**

| Metric | Interpretation |
|---|---|
| ARI (Adjusted Rand Index) | Cluster assignment quality; 1 = perfect, 0 = random |
| π MAE | Error in inferred cluster proportions |
| μ KL | Bernoulli KL divergence of mutation profiles |

---

## Output Files

The pipeline writes to `<output_dir>/<sample>_<ref>_<start>_<end>/`:

| File | Description |
|---|---|
| `K_<k>/log_likelihoods.txt` | ELBO per run |
| `K_<k>/Largest_LogLike.txt` | Best ELBO across runs |
| `K_<k>/BIC.txt` | BIC score used for K selection |
| `K_<k>/Clusters_Mu.txt` | Inferred μ per cluster (main result) |
| `K_<k>/Responsibilities.txt` | Soft cluster assignments q_nk |
| `K_<k>/Proportions.txt` | Inferred π per cluster |
| `K_<k>/plots/*.html` | Interactive plotly visualisations |
| `log.txt` | Run summary log |

---

## Dependencies

| Package | Purpose |
|---|---|
| numpy | Core numerics |
| scipy | digamma, logsumexp, gammaln, optimisation |
| scikit-learn | Warm-start init (BayesianGaussianMixture), ARI metric |
| pandas | Data I/O |
| matplotlib | Benchmark plots |
| plotly | Interactive pipeline output plots |
| bowtie2 | Read alignment (pipeline only) |
| RNAstructure | RNA folding (optional, `--struct` mode) |






