# VARISEM
VARISEM leverages long-read sequencing to uncover and resolve alternative RNA structures by clustering single-molecule mutation profiles. It provides deep insights into RNA structural heterogeneity, enabling the exploration of complex RNA conformational ensembles and their modulation by chemical modifications.

## Table of Contents
- [Background](#background)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Input Data](#input-data)
- [Quick Start (Basic Usage)](#quick-start-basic-usage)
- [Output](#output)
- [Advanced Settings](#advanced-settings)
- [Examples and Notebooks](#examples-and-notebooks)
- [Contact](#contact)

## Background
RNA molecules exhibit significant structural heterogeneity, further complicated by dynamic chemical modifications such as **N6-methyladenosine (m6A)**. This variability presents a major challenge for traditional single-structure prediction methods. Although SHAPE-MaP provides experimental constraints that improve RNA structure prediction, existing computational approaches vary in their ability to model complex RNA conformational ensembles.
For example, RNAfold2 predicts RNA structures based on thermodynamic energy parameters and can explore alternative conformations through sampling. However, it does not explicitly model the probabilistic distribution of coexisting conformations within a heterogeneous RNA population.
### Why VARISEM?
To address these limitations, we developed **VARISEM**—a **Variational Bayesian clustering framework** designed to infer **RNA structural ensembles** from high-throughput **SHAPE-MaP** and **eTAM-seq data**. By integrating structural and modification-specific signals at the single-molecule level, VARISEM enables quantitative analysis of how m6A modifications reshape RNA structural landscapes, such as that of **7SK RNA**.

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
VARISEM works on binary datasets.
* Input `X` should be a NumPy array or Pandas DataFrame.
* Rows represent samples (N samples).
* Columns represent features (D binary features, e.g., mutation presence/absence).
You can also explore our simualated data under `Simulated Data`folder

### Quick Start (Basic Usage)
You can recall `VARISEM function` by running 
```python
from varisem import VARISEM
from varisem import VARISEM

# Initialize the model
VARISEM_model = VARISEM(X, K=3, max_iter=300, tol=1e-6)

# Fit the model
VARISEM_model.fit()

# Extract inferred cluster proportions
pi_inferred_mean = VARISEM_model.pi_k_mean

# Extract inferred mutation probabilities for each cluster
mu_inferred_mean = VARISEM_model.mu_ki_mean

# Predict cluster assignments for each molecule
cluster_assignments = VARISEM_model.predict()

```

### Output
After running `VARISEM_model.fit()`, you can extract:
* `pi_inferred_mean`: the estimated cluster proportions (array of size K).
* `mu_inferred_mean`: the estimated feature probabilities for each cluster (array of size K x D).
* `cluster_assignments`: a list of the predicted cluster index for each sample (N samples).

### Advanced Settings
VARISEM supports several customizable parameters when initializing the model:
* `K`: Number of clusters (components).
* `max_iter`: Maximum iterations for variational optimization (default: 300).
* `tolv: Tolerance for convergence (default: 1e-6).
* `alpha_prior`: Prior for cluster proportions (Dirichlet).
* `a_prior`, `b_prior`: Priors for Beta distributions over feature probabilities.

### Examples and Notebooks
Explore the `Jupyter Notebook` folder for:
* Tutorials on simulate and preprocessing data.
* Example workflows demonstrating how to run VARISEM.
* Visualizations of clustering results (ARI, MAE and KL)  and estimated distributions against the ground truth. 

### Contact
For questions, issues, or feature requests, feel free to [open an issue]([https://github.com/QSong-github/VARISEM/issues) or contact me directly.  
If you use VARISEM in your research, please cite:
`VARISEM: Variational Bayesian Clustering for RNA Structural Ensembles`.





