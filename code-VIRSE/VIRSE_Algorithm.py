#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import os
from VARISEM import VIRSE


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
        Number of threads (not used in VI, kept for interface compatibility)

    Returns
    -------
    (elbo_list, final_mu, final_obs_pi, final_real_pi, resps, BIC)
    or None if collapse detected.
    """

    # ======= Extract data =======
    X_mat = np.asarray(X.BV_Matrix, dtype=float)
    N, D = X_mat.shape

    seed = int(os.getenv("VI_SEED", np.random.randint(1, 1_000_000)))
    print(f"[VIRSE] Starting variational inference for K={K} (seed={seed})")

    # ======= Initialize VIRSE model with biological priors =======
    vi = VIRSE(
        X=X_mat,
        K=K,
        max_iter=max(10, MIN_ITS),
        tol=CONV_CUTOFF,
        alpha_prior=np.ones(K) * 0.3,      # sparse Dirichlet prior
        a_prior=np.ones((K, D)) * 1.0,     # low-reactivity Beta prior
        b_prior=np.ones((K, D)) * 1.0,
        random_state=seed,
    )

    vi.fit(verbose_every=10)
    res = vi.get_results()

    pi       = res["pi_inferred_mean"]
    mu       = res["mu_inferred_mean"]
    resps    = res["q_nk"]
    elbo_list = res["elbo_values"]

    # ======= Collapse detection (skip for K=1) =======
    if K > 1 and (np.any(pi < 0.01) or np.any(pi > 0.99)):
        print(f"[WARN] Collapse detected for K={K}: π̂={np.round(pi,3)} → Skipping this run.")
        return None

    # ======= Symmetry detection and re-run with asymmetric priors =======
    if K > 1 and np.allclose(pi, np.ones(K) / K, atol=0.05):
        print(f"[WARN] Symmetric π detected for K={K}: {np.round(pi,3)} → Re-running with asymmetric priors.")
        vi = VIRSE(
            X=X_mat,
            K=K,
            max_iter=max(10, MIN_ITS),
            tol=CONV_CUTOFF,
            alpha_prior=np.ones(K) * 0.3,
            a_prior=np.ones((K, D)) * 1.0,
            b_prior=np.ones((K, D)) * 1.0,
            random_state=seed + 1,
        )
        vi.fit(verbose_every=10)
        res       = vi.get_results()
        pi        = res["pi_inferred_mean"]
        mu        = res["mu_inferred_mean"]
        resps     = res["q_nk"]
        elbo_list = res["elbo_values"]

    # ======= VBIC calculation =======
    p = K * D
    last_elbo = elbo_list[-1] if len(elbo_list) else -np.inf
    BIC = np.log(max(N, 1)) * p - 2 * last_elbo

    # ======= Diagnostics output =======
    mu_diff = np.mean(np.abs(mu[0] - mu[1])) if K == 2 else np.nan
    print(f"[VIRSE] Finished K={K}: π̂={np.round(pi,3)}, mean μ diff={mu_diff:.4f}, VBIC={BIC:.4f}")

    return (elbo_list, mu, pi, pi, resps, BIC)

