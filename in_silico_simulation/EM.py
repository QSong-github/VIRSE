#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# VIRSE — Variational Inference for RNA Structure Ensembles
"""
Pure Expectation–Maximization (EM) Clustering
---------------------------------------------
This version preserves the numerical tricks from the provided DREEM class
(logsumexp + clipping), but uses the same interface as the original PureEM
used in the VARISEM/DRACO simulation pipeline.
"""

import numpy as np
import pandas as pd
from scipy.special import logsumexp


class PureEM:
    def __init__(self, X, K, MIN_ITS=20, CONV_CUTOFF=1e-4, CPUS=4, random_state=42):
        self.X = np.asarray(X, dtype=float)
        self.N, self.D = self.X.shape
        self.K = K
        self.MIN_ITS = MIN_ITS
        self.CONV_CUTOFF = CONV_CUTOFF
        self.random_state = np.random.RandomState(random_state)

        # parameters
        self.pi = np.ones(K) / K
        self.mu = np.random.rand(K, self.D)
        self.resps = np.zeros((self.N, self.K))
        self.log_like_list = []


    # ======================================================
    # E-step
    # ======================================================
    def e_step(self):
        log_likelihood = np.zeros((self.N, self.K))

        for k in range(self.K):
            log_likelihood[:, k] = (
                np.sum(
                    self.X * np.log(np.clip(self.mu[k], 1e-10, 1 - 1e-10)) +
                    (1 - self.X) * np.log(np.clip(1 - self.mu[k], 1e-10, 1 - 1e-10)),
                    axis=1
                )
                + np.log(self.pi[k] + 1e-10)
            )

        # log-resps = log p(x|k) + log pi_k - logsumexp
        log_resps = log_likelihood - logsumexp(log_likelihood, axis=1, keepdims=True)
        self.resps = np.exp(log_resps)


    # ======================================================
    # M-step
    # ======================================================
    def m_step(self):
        N_k = np.sum(self.resps, axis=0)

        # update pi
        self.pi = N_k / self.N

        # update mu
        for k in range(self.K):
            self.mu[k] = (
                np.sum(self.resps[:, k][:, None] * self.X, axis=0)
                / (N_k[k] + 1e-10)
            )


    # ======================================================
    # Log-likelihood
    # ======================================================
    def compute_log_likelihood(self):
        log_likelihood = np.zeros((self.N, self.K))

        for k in range(self.K):
            log_likelihood[:, k] = (
                np.sum(
                    self.X * np.log(np.clip(self.mu[k], 1e-10, 1 - 1e-10)) +
                    (1 - self.X) * np.log(np.clip(1 - self.mu[k], 1e-10, 1 - 1e-10)),
                    axis=1
                )
                + np.log(self.pi[k] + 1e-10)
            )

        return np.sum(logsumexp(log_likelihood, axis=1))


    # ======================================================
    # FIT
    # ======================================================
    def fit(self):
        print(f"[EM] Starting PureEM (DREEM-style) K={self.K}, N={self.N}, D={self.D}")

        log_old = -np.inf
        iteration = 0

        while True:
            self.e_step()
            self.m_step()

            log_l = self.compute_log_likelihood()
            self.log_like_list.append(log_l)

            if iteration >= self.MIN_ITS:
                if abs(log_l - log_old) < self.CONV_CUTOFF:
                    print(f"[EM] Converged after {iteration} iterations.")
                    break

            log_old = log_l
            iteration += 1

            if iteration >= 1000:
                print("[WARN] EM Max iterations reached.")
                break

        self.final_mu = self.mu
        self.final_pi = self.pi
        self.n_iter = iteration


    # ======================================================
    # SAVE RESULTS
    # ======================================================
    def save_results(self, pi_filename, mu_filename, cluster_filename):
        clusters = np.argmax(self.resps, axis=1)

        pd.DataFrame(
            [self.final_pi],
            columns=[f"Cluster_{k+1}" for k in range(self.K)]
        ).to_csv(pi_filename, index=False)

        pd.DataFrame(self.final_mu).to_csv(mu_filename, index=False)
        pd.DataFrame({"Cluster": clusters}).to_csv(cluster_filename, index=False)

        print("EM results saved successfully.")
