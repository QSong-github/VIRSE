#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from VARISEM import VARISEM


def Run_EM(X, K, MIN_ITS, CONV_CUTOFF, CPUS):
    """
    Replacement for classical EM using Variational Inference (VI).

    Parameters
    ----------
    X : object
        Data object with attributes:
          - BV_Matrix: binary or fractional matrix (N x D)
          - BV_Abundance: sample weights (N,)
    K : int
        Number of clusters
    MIN_ITS : int
        Minimum number of VI iterations
    CONV_CUTOFF : float
        Convergence threshold for ELBO
    CPUS : int
        Number of threads (not used in VI yet, for compatibility)

    Returns
    -------
    (elbo_list, final_mu, final_obs_pi, final_real_pi, resps, BIC)
    or None if collapse detected.
    """

    # ======= Extract data =======
    X_mat = np.asarray(X.BV_Matrix, dtype=float)
    weights = np.asarray(X.BV_Abundance, dtype=float)
    N, D = X_mat.shape

    seed = int(os.getenv("VI_SEED", np.random.randint(1, 1_000_000)))
    print(f"[VI] Starting variational inference for K={K} (seed={seed})")

    # ======= Adaptive regularization =======
    reg = 1e-3 if N < 50000 else 1e-3

    # ======= Initialize VI model with biological priors =======
    vi = VARISEM(
        X=X_mat,
        K=K,
        max_iter=max(10, MIN_ITS),
        tol=CONV_CUTOFF,
        alpha_prior=np.ones(K) * 0.3,           # sparse Dirichlet prior
        a_prior=np.ones((K, D)) * 1,          # low-reactivity Beta prior
        b_prior=np.ones((K, D)) * 1,
        sample_weights=weights,
        random_state=seed
    )

    vi._fit_sparse(verbose_every=10, rank=30, reg=reg)
    res = vi.get_results()

    pi = res["pi_inferred_mean"]
    mu = res["mu_inferred_mean"]
    resps = res["q_nk"]
    elbo_list = res["elbo_values"]

    # ======= Collapse detection (skip for K=1) =======
    if K > 1 and (np.any(pi < 0.01) or np.any(pi > 0.99)):
        print(f"[WARN] Collapse detected for K={K}: π̂={np.round(pi,3)} → Skipping this run.")
        return None

    # ======= Symmetry detection and re-run with asymmetric priors =======
    if K > 1 and np.allclose(pi, np.ones(K) / K, atol=0.05):
        print(f"[WARN] Symmetric π detected for K={K}: {np.round(pi,3)} → Re-running with asymmetric priors.")
        vi = VARISEM(
            X=X_mat,
            K=K,
            max_iter=max(10, MIN_ITS),
            tol=CONV_CUTOFF,
            alpha_prior=np.ones(K) * 0.3,
            a_prior=np.ones((K, D)) * 1,
            b_prior=np.ones((K, D)) * 1,
            sample_weights=weights,
            random_state=seed + 1
        )
        vi._fit_sparse(verbose_every=10, rank=30, reg=1e-3)
        res = vi.get_results()
        pi = res["pi_inferred_mean"]
        mu = res["mu_inferred_mean"]
        resps = res["q_nk"]
        elbo_list = res["elbo_values"]

    # ======= VBIC calculation =======
    p = K * D
    last_elbo = elbo_list[-1] if len(elbo_list) else -np.inf
    BIC = np.log(max(N, 1)) * p - 2 * last_elbo

    # ======= Diagnostics output =======
    mu_diff = np.mean(np.abs(mu[0] - mu[1])) if K == 2 else np.nan
    print(f"[VI] Finished K={K}: π̂={np.round(pi,3)}, mean μ diff={mu_diff:.4f}, VBIC={BIC:.4f}")

    return (elbo_list, mu, pi, pi, resps, BIC)

