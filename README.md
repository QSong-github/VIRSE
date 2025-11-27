# VIRSE
VIRSE leverages long-read sequencing to uncover and resolve alternative RNA structures by clustering single-molecule mutation profiles. It provides deep insights into RNA structural heterogeneity, enabling the exploration of complex RNA conformational ensembles and their modulation by chemical modifications.

## Table of Contents
- [Background](#background)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Input Data](#input-data)
- [Quick Start (Basic Usage)](#quick-start-basic-usage)
- [Output](#output)
- [Advanced Settings](#advanced-settings)
- [Examples and Notebooks](#examples-and-notebooks)
- [In Silico Simulation Module](#in-silico-simulation-module)
- [Processing Real DMS-MaPseq Data](#processing-real-dms-mapseq-data)
- [Contact](#contact)
  

## Background
RNA molecules exhibit significant structural heterogeneity, further complicated by dynamic chemical modifications such as N6-methyladenosine (m6A). This variability presents a major challenge for traditional single-structure prediction methods. Although SHAPE-MaP provides experimental constraints that improve RNA structure prediction, existing computational approaches vary in their ability to model complex RNA conformational ensembles.
For example, RNAfold2 predicts RNA structures based on thermodynamic energy parameters and can explore alternative conformations through sampling. However, it does not explicitly model the probabilistic distribution of coexisting conformations within a heterogeneous RNA population.
### Why VIRSE?
To address these limitations, we developed VIRSE designed to infer RNA structural ensembles** from high-throughput SHAPE-MaP and eTAM-seq data. By integrating structural and modification-specific signals at the single-molecule level, VIRSE enables quantitative analysis of how m6A modifications reshape RNA structural landscapes, such as that of 7SK RNA.

## Getting Started
### Prerequisites
You need Python 3.7+ and the following libraries:
* numpy
* pandas
* scipy
* scikit-learn
* matplotlib (optional for visualization)
Install the required packages using pip:
```python
pip install numpy pandas scipy scikit-learn matplotlib
```

### Input Data
VIRSE works on binary datasets.
* Input `X` should be a NumPy array or Pandas DataFrame.
* Rows represent samples (N samples).
* Columns represent features (D binary features, e.g., mutation presence/absence).
You can also explore our simualated data under `Simulated Data`folder

### Quick Start (Basic Usage)
You can recall `VIRSE function` by running 
```python
from VIRSE import VIRSE
from VIRSE import VIRSE

# Initialize the model
VIRSE_model = VIRSE(X, K=3, max_iter=300, tol=1e-6)

# Fit the model
VIRSE_model.fit()

# Extract inferred cluster proportions
pi_inferred_mean = VIRSE_model.pi_k_mean

# Extract inferred mutation probabilities for each cluster
mu_inferred_mean = VIRSE_model.mu_ki_mean

# Predict cluster assignments for each molecule
cluster_assignments = VIRSE_model.predict()

```

### Output
After running `VIRSE_model.fit()`, you can extract:
* `pi_inferred_mean`: the estimated cluster proportions (array of size K).
* `mu_inferred_mean`: the estimated feature probabilities for each cluster (array of size K x D).
* `cluster_assignments`: a list of the predicted cluster index for each sample (N samples).

### Advanced Settings
VIRSE supports several customizable parameters when initializing the model:
* `K`: Number of clusters (components).
* `max_iter`: Maximum iterations for variational optimization (default: 300).
* `tolv: Tolerance for convergence (default: 1e-6).
* `alpha_prior`: Prior for cluster proportions (Dirichlet).
* `a_prior`, `b_prior`: Priors for Beta distributions over feature probabilities.

### Examples and Notebooks
Explore the `Jupyter Notebook` folder for:
* Tutorials on simulate and preprocessing data.
* Example workflows demonstrating how to run VIRSE.
* Visualizations of clustering results (ARI, MAE and KL)  and estimated distributions against the ground truth.
   
<img width="658" alt="Screenshot 2025-03-21 at 7 31 36 PM" src="https://github.com/user-attachments/assets/867aa7c8-1538-41c2-9733-956666dafbf6" />

### In Silico Simulation Module
This module provides a fully controllable in silico simulation framework for generating DRACO-style single-molecule mutation data. The simulator produces synthetic DMS-MaPseq–like mutation matrices with user-specified structural patterns, mutation rates, background noise, cluster proportions, and sequencing depth.

It is used to benchmark VIRSE, EM, and other clustering or mixture-model algorithms under realistic yet fully known ground truth.
### Processing Real DMS-MaPseq Data

`code-VIRSE` supports direct processing of real DMS-MaPseq datasets using a workflow compatible with the CodeOcean DRACO/DREEM capsule.

This functionality is adapted from:

**CodeOcean Capsule 6175523**  
https://codeocean.com/capsule/6175523/tree/v1

with the following updated components:

- `VIRSE.py` — replaced with the updated Variational Bayesian inference model  
- `EM_algorithm.py` — improved and stabilized EM implementation  

All other scripts follow the original DRACO/DREEM folder structure.

---

### Pipeline Overview

The script `Run_DREEM.py` performs:

- FASTQ mutation extraction  
- Bit-vector construction  
- Clustering using updated EM or VIRSE  
- Folding cluster-specific structures using RNAstructure (optional)

---

### Quick Start Example

To process the demo dataset `RRE_invitroDMS`:

```bash
python Run_DREEM.py \
    ../data/DREEM_Input/ \
    ../results/ \
    RRE_invitroDMS \
    NL43rna \
    7410 7500 \
    --fastq --struct
```

Change the parameters based on your file format.

### Contact
For questions, issues, or feature requests, feel free to [open an issue]([https://github.com/QSong-github/VIRSE/issues) or contact me directly.  
If you use VIRSE in your research, please cite:
`VIRSE: Variational Bayesian Clustering for RNA Structural Ensembles`.





