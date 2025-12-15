#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VARISEM v3.1
-------------------------------------------
v3 stable algorithm + v2.9 interface

Key properties:
- μ ALWAYS from Beta posterior (no overwrite)
- mu_tmp only used for R/W gradient
- No collapse (π=001, KL爆炸, ARI=0 cases disappear)
- Compatible with original VARISEM v2.9 API
"""

import numpy as np
from scipy.special import digamma, logsumexp, gammaln
from autograd import grad
from autograd import numpy as anp


class VARISEM:

    def __init__(self, X, K, max_iter=300, tol=1e-6,
                 alpha_prior=None, a_prior=None, b_prior=None,
                 sample_weights=None, random_state=42):

        # Data
        self.X = np.asarray(X, float)
        self.N, self.D = self.X.shape
        self.K = int(K)
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.RandomState(random_state)

        self.w = (np.ones((self.N, 1))
                  if sample_weights is None else
                  np.asarray(sample_weights).reshape(-1, 1))

        # ===== Priors (same as v3) =====
        self.alpha_prior = np.ones(self.K) * 0.5 if alpha_prior is None else np.asarray(alpha_prior)
        self.a_prior = np.ones((self.K, self.D)) * 1.5 if a_prior is None else np.asarray(a_prior)
        self.b_prior = np.ones((self.K, self.D)) * 10 if b_prior is None else np.asarray(b_prior)

        # ===== Initialization =====
        self.q_nk = self.rng.dirichlet(np.ones(self.K), size=self.N)
        self.q_nk += self.rng.normal(0, 0.01, self.q_nk.shape)
        self.q_nk = np.clip(self.q_nk, 1e-6, None)
        self.q_nk /= np.sum(self.q_nk, axis=1, keepdims=True)

        WQ = (self.q_nk.T * self.w.T)

        self.alpha_q = self.alpha_prior + np.sum(self.w * self.q_nk, axis=0)
        self.a_q = self.a_prior + WQ @ self.X
        self.b_q = self.b_prior + WQ @ (1 - self.X)

        # Mean parameters
        self.pi_inferred_mean = self.alpha_q / np.sum(self.alpha_q)
        self.mu_inferred_mean = self.a_q / (self.a_q + self.b_q)

        self.cluster_assignments = None
        self.elbo_values = []


    # ------------------------------
    #      Helper expectations
    # ------------------------------
    def _elog_pi(self):
        return digamma(self.alpha_q) - digamma(np.sum(self.alpha_q))

    def _elog_mu(self):
        digsum = digamma(self.a_q + self.b_q)
        return digamma(self.a_q) - digsum, digamma(self.b_q) - digsum


    # ------------------------------
    #        Update q_nk
    # ------------------------------
    def update_q_nk(self):

        e_log_pi = self._elog_pi()[None, :]
        e_log_mu, e_log_1mu = self._elog_mu()

        log_q = (
            self.X[:, None, :] * e_log_mu[None, :, :] +
            (1 - self.X)[:, None, :] * e_log_1mu[None, :, :]
        ).sum(axis=2)

        log_q += e_log_pi
        log_q -= logsumexp(log_q, axis=1, keepdims=True)

        self.q_nk = np.exp(log_q)
        self.q_nk = np.nan_to_num(self.q_nk, nan=1/self.K)


    # ------------------------------
    #        Update π
    # ------------------------------
    def update_pi(self):
        self.alpha_q = self.alpha_prior + np.sum(self.w * self.q_nk, axis=0)
        self.pi_inferred_mean = self.alpha_q / np.sum(self.alpha_q)


    # ------------------------------
    #   Update μ ONLY from Beta posterior
    # ------------------------------
    def update_mu_from_posterior(self):
        WQ = (self.q_nk.T * self.w.T)
        self.a_q = self.a_prior + WQ @ self.X
        self.b_q = self.b_prior + WQ @ (1 - self.X)

        self.mu_inferred_mean = np.nan_to_num(
            self.a_q / (self.a_q + self.b_q),
            nan=1e-6, posinf=1-1e-6, neginf=1e-6
        )


    # ------------------------------
    #        ELBO computation
    # ------------------------------
    def compute_elbo(self):
        e_log_mu, e_log_1mu = self._elog_mu()

        like_nk = (
            self.X[:, None, :] * e_log_mu[None, :, :] +
            (1 - self.X)[:, None, :] * e_log_1mu[None, :, :]
        ).sum(axis=2)

        E_log_pX = np.sum(self.w * np.sum(self.q_nk * like_nk, axis=1))
        e_log_pi = self._elog_pi()
        E_log_pZ = np.sum(self.w * np.sum(self.q_nk * e_log_pi, axis=1))

        # Dirichlet prior
        logB_prior = np.sum(gammaln(self.alpha_prior)) - gammaln(np.sum(self.alpha_prior))
        E_log_ppi = -logB_prior + np.sum((self.alpha_prior - 1) * e_log_pi)

        # Beta priors
        logB_ab0 = gammaln(self.a_prior) + gammaln(self.b_prior) - gammaln(self.a_prior + self.b_prior)
        E_log_pmu = -np.sum(logB_ab0) + np.sum(
            (self.a_prior - 1) * e_log_mu + (self.b_prior - 1) * e_log_1mu
        )

        H_qZ = -np.sum(self.w * np.sum(self.q_nk * np.log(self.q_nk + 1e-10), axis=1))

        alpha0 = np.sum(self.alpha_q)
        logB_alpha = np.sum(gammaln(self.alpha_q)) - gammaln(alpha0)
        H_qpi = (logB_alpha + (alpha0 - self.K) * digamma(alpha0)
                 - np.sum((self.alpha_q - 1) * digamma(self.alpha_q)))

        H_qmu = np.sum(
            gammaln(self.a_q + self.b_q)
            - gammaln(self.a_q) - gammaln(self.b_q)
            + (self.a_q - 1) * digamma(self.a_q)
            + (self.b_q - 1) * digamma(self.b_q)
            - (self.a_q + self.b_q - 2) * digamma(self.a_q + self.b_q)
        )

        return float(E_log_pX + E_log_pZ + E_log_ppi + E_log_pmu + H_qZ + H_qpi + H_qmu)


    # ------------------------------
    #     Sparse VI (v3 kernel)
    # ------------------------------
    def _fit_sparse(self, rank=10, reg=1e-4, lr=5e-3,
                    momentum=0.9, clip_val=5.0, verbose_every=10):

        print(f"[Sparse VI] rank={rank}, reg={reg}")

        r = min(rank, self.D)
        R = self.rng.randn(self.K, r) * 0.1
        W = self.rng.randn(self.D, r) * 0.1

        vR = np.zeros_like(R)
        vW = np.zeros_like(W)

        elbo_prev = -np.inf

        def safe_sigmoid(z):
            z = anp.clip(z, -20, 20)
            return 1 / (1 + anp.exp(-z))

        def elbo_autograd(R_, W_):
            z = anp.clip(R_ @ W_.T, -20, 20)
            mu_tmp = safe_sigmoid(z)
            mu_tmp = anp.clip(mu_tmp, 1e-6, 1-1e-6)

            e_log_mu = anp.log(mu_tmp)
            e_log_1mu = anp.log(1 - mu_tmp)

            like = anp.dot(self.X, e_log_mu.T) + anp.dot(1 - self.X, e_log_1mu.T)
            like = anp.sum(self.w * anp.sum(self.q_nk * like, axis=1))

            return like - 0.5 * reg * (anp.sum(R_**2) + anp.sum(W_**2))

        # iteration
        for it in range(self.max_iter):

            # 1) Low rank μ_tmp (only used for gradient)
            mu_tmp = 1 / (1 + np.exp(-np.clip(R @ W.T, -20, 20)))

            # 2) True μ from Beta posterior (stability)
            self.update_mu_from_posterior()

            # 3) Update q_nk & π
            self.update_q_nk()
            self.update_pi()

            # 4) Gradient for R/W
            grad_R = grad(lambda X_: -elbo_autograd(X_, W))(R)
            grad_W = grad(lambda X_: -elbo_autograd(R, X_))(W)

            grad_R = np.clip(grad_R, -clip_val, clip_val)
            grad_W = np.clip(grad_W, -clip_val, clip_val)

            # Momentum-SGD
            vR = momentum * vR + lr * grad_R
            vW = momentum * vW + lr * grad_W
            R -= vR
            W -= vW

            # 5) ELBO
            elbo = self.compute_elbo()
            self.elbo_values.append(elbo)

            if abs(elbo - elbo_prev) < self.tol:
                print(f"[Sparse VI] Converged at {it}, ELBO={elbo:.4f}")
                break

            if it % verbose_every == 0:
                print(f"[Sparse VI] Iter {it:4d} | ELBO={elbo:.4f}")

            elbo_prev = elbo

        self.cluster_assignments = np.argmax(self.q_nk, axis=1)
        print("[Sparse VI] Finished.")


    # ------------------------------
    #         Public fit()
    # ------------------------------
    def fit(self):
        print(f"[VARISEM v3.1] Running, K={self.K}")
        self._fit_sparse()
        print(f"[VARISEM v3.1] Done. π̂ = {np.round(self.pi_inferred_mean, 3)}")


    # ------------------------------
    #        Return results
    # ------------------------------
    def get_results(self):
        return {
            "pi_inferred_mean": self.pi_inferred_mean,
            "mu_inferred_mean": self.mu_inferred_mean,
            "cluster_assignments": np.argmax(self.q_nk, axis=1),
            "q_nk": self.q_nk,
            "elbo_values": self.elbo_values
        }
