# VARISEM
VARISEM leverages long-read sequencing to uncover and resolve alternative RNA structures by clustering single-molecule mutation profiles, providing deep insights into RNA structural heterogeneity. 

## Table of Contents

## Background
RNA molecules exhibit significant structural heterogeneity, further complicated by dynamic chemical modifications such as **N6-methyladenosine (m6A)**. This variability presents a major challenge for traditional single-structure prediction methods. Although SHAPE-MaP provides experimental constraints that improve RNA structure prediction, existing computational approaches vary in their ability to model complex RNA conformational ensembles.

For example, RNAfold2 predicts RNA structures based on thermodynamic energy parameters and can explore alternative conformations through sampling. However, it does not explicitly model the probabilistic distribution of coexisting conformations within a heterogeneous RNA population.

To address these limitations, we developed VARISEM, a **Variational Bayesian clustering framework** designed to infer **RNA structural ensembles** from high-throughput **SHAPE-MaP** and **eTAM-seq** data. By integrating both structural information and modification-specific signals at the single-molecule level, VARISEM enables quantitative analysis of how m6A modifications reshape the structural landscape of 7SK RNA.

## Guided Tutorial
### Requirement
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import digamma, logsumexp
from sklearn.mixture import BayesianGaussianMixture
from sklearn.metrics import adjusted_rand_score
from scipy.special import digamma, logsumexp, gammaln
```
### Input Data
A list binary
examples of input_data:

You can also use our simualated data under `Simulated Data`folder
### Command:
You can recall `VARISEM function` by 
```python
vi_model = VARISEM(X, K=K, max_iter=300, tol=1e-6)
vi_model.fit()
pi_inferred_mean = vi_model.pi_k_mean
mu_inferred_mean = vi_model.mu_ki_mean
```
### Output


For more details or different experience setting, Please check `Jupter Notebook`folder




