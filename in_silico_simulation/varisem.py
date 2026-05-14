#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# VIRSE — Variational Inference for RNA Structure Ensembles
"""
Three inference algorithms for the Bayesian Bernoulli mixture model:

  VIRSE     — Mean-approx variational Bayes (the default VIRSE model).
  VIRSE_VI  — Standard conjugate VB-EM with true ELBO.
  Gibbs     — Blocked Gibbs sampler (full Bayesian MCMC).

Model
-----
  pi ~ Dir(alpha)          alpha_prior = 10 per component
  mu_ki ~ Beta(a, b)       a_prior = 5, b_prior = 10 per feature
  z_n ~ Cat(pi)
  x_ni | z_n=k ~ Bernoulli(mu_ki)
"""

import numpy as np
import pandas as pd
from scipy.special import digamma, logsumexp, gammaln
from sklearn.mixture import BayesianGaussianMixture


# ---------------------------------------------------------------------------
# Shared warm-start helper
# ---------------------------------------------------------------------------

def _bgm_init(X, K, random_state=42):
    bgm = BayesianGaussianMixture(
        n_components=K,
        covariance_type="full",
        weight_concentration_prior_type="dirichlet_process",
        random_state=random_state,
        max_iter=100,
        n_init=10,
    )
    bgm.fit(X)
    return bgm.predict_proba(X)  # (N, K)


# ===========================================================================
#  VIRSE — mean-approx variational Bayes
# ===========================================================================

class VIRSE:
    """
    Mean-based variational approximation for the Bayesian Bernoulli mixture.

    q_nk ∝ pi_k · Π_i  mu_ki^x_ni · (1-mu_ki)^(1-x_ni)

    Uses posterior means (not digamma expectations) for pi_k and mu_ki.
    """

    def __init__(self, X, K, max_iter=200, tol=1e-6,
                 alpha_prior=None, a_prior=None, b_prior=None,
                 random_state=42):
        self.X = np.asarray(X, dtype=float)
        self.N, self.D = self.X.shape
        self.K = int(K)
        self.max_iter = max_iter
        self.tol = tol

        self.alpha_prior = np.ones(self.K) * 10.0  if alpha_prior is None else np.asarray(alpha_prior, float)
        self.a_prior     = np.ones((self.K, self.D)) * 5.0  if a_prior is None else np.asarray(a_prior, float)
        self.b_prior     = np.ones((self.K, self.D)) * 10.0 if b_prior is None else np.asarray(b_prior, float)

        self.q_nk   = _bgm_init(self.X, self.K, random_state)
        self.alpha_q = self.alpha_prior + np.sum(self.q_nk, axis=0)
        self.a_q     = self.a_prior + self.q_nk.T @ self.X
        self.b_q     = self.b_prior + self.q_nk.T @ (1.0 - self.X)

        self.pi_inferred_mean = self.alpha_q / np.sum(self.alpha_q)
        self.mu_inferred_mean = self.a_q / (self.a_q + self.b_q)
        self.elbo_values      = []
        self.cluster_assignments = None

    def update_q_nk(self):
        eps = 1e-12
        pi = np.clip(self.pi_inferred_mean, eps, 1.0 - eps)
        mu = np.clip(self.mu_inferred_mean, eps, 1.0 - eps)
        log_q = (np.log(pi)[None, :]
                 + np.einsum("nd,kd->nk", self.X,         np.log(mu))
                 + np.einsum("nd,kd->nk", 1.0 - self.X,  np.log(1.0 - mu)))
        log_q -= logsumexp(log_q, axis=1, keepdims=True)
        self.q_nk = np.exp(log_q)

    def update_pi(self):
        self.alpha_q = self.alpha_prior + np.sum(self.q_nk, axis=0)
        self.pi_inferred_mean = self.alpha_q / np.sum(self.alpha_q)

    def update_mu(self):
        self.a_q = self.a_prior + self.q_nk.T @ self.X
        self.b_q = self.b_prior + self.q_nk.T @ (1.0 - self.X)
        self.mu_inferred_mean = self.a_q / (self.a_q + self.b_q)

    def compute_elbo(self):
        eps = 1e-12
        pi = np.clip(self.pi_inferred_mean, eps, 1.0 - eps)
        mu = np.clip(self.mu_inferred_mean, eps, 1.0 - eps)
        ll  = np.sum(self.q_nk[:, :, None] * (
                  self.X[:, None, :] * np.log(mu)[None, :, :]
                + (1.0 - self.X[:, None, :]) * np.log(1.0 - mu)[None, :, :]))
        lz  = np.sum(self.q_nk * np.log(pi))
        lp  = np.sum((self.alpha_prior - 1.0) * np.log(pi))
        lpm = np.sum((self.a_prior - 1.0) * np.log(mu) + (self.b_prior - 1.0) * np.log(1.0 - mu))
        hz  = -np.sum(self.q_nk * np.log(self.q_nk + eps))
        return float(ll + lz + lp + lpm + hz)

    def fit(self, verbose_every=10):
        score_old = -np.inf
        self.elbo_values = []
        for it in range(self.max_iter):
            self.update_q_nk()
            self.update_pi()
            self.update_mu()
            score = self.compute_elbo()
            self.elbo_values.append(score)
            if abs(score - score_old) < self.tol:
                if verbose_every:
                    print(f"[VIRSE] Converged at iter {it}.")
                break
            score_old = score
            if verbose_every and it % verbose_every == 0:
                print(f"[VIRSE] iter={it:4d}  score={score:.4f}")
        self.cluster_assignments = np.argmax(self.q_nk, axis=1)
        return self

    def predict(self):
        if self.cluster_assignments is None:
            raise ValueError("Run fit() first.")
        return self.cluster_assignments

    def get_results(self):
        return {
            "pi_inferred_mean":    self.pi_inferred_mean,
            "mu_inferred_mean":    self.mu_inferred_mean,
            "cluster_assignments": self.cluster_assignments,
            "q_nk":                self.q_nk,
            "elbo_values":         self.elbo_values,
        }

    def save_results(self, out_dir='.'):
        import os; os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame([self.pi_inferred_mean],
                     columns=[f'K{k}' for k in range(self.K)]).to_csv(
            os.path.join(out_dir, 'VIRSE_pi.csv'), index=False)
        pd.DataFrame(self.mu_inferred_mean).to_csv(
            os.path.join(out_dir, 'VIRSE_mu.csv'), index=False)
        pd.DataFrame({'cluster': self.cluster_assignments}).to_csv(
            os.path.join(out_dir, 'VIRSE_assignments.csv'), index=False)


# ===========================================================================
#  VIRSE_VI — standard conjugate VB-EM (true ELBO)
# ===========================================================================

class VIRSE_VI:
    """
    Standard conjugate variational Bayes EM.

    q_nk uses E[log pi_k] and E[log mu_ki] computed via digamma,
    and optimises the true ELBO (including Dirichlet and Beta entropies).
    """

    def __init__(self, X, K, max_iter=300, tol=1e-6,
                 alpha_prior=None, a_prior=None, b_prior=None,
                 random_state=42):
        self.X = np.asarray(X, dtype=float)
        self.N, self.D = self.X.shape
        self.K = int(K)
        self.max_iter = max_iter
        self.tol = tol

        self.alpha_prior = np.ones(self.K) * 10.0  if alpha_prior is None else np.asarray(alpha_prior, float)
        self.a_prior     = np.ones((self.K, self.D)) * 5.0  if a_prior is None else np.asarray(a_prior, float)
        self.b_prior     = np.ones((self.K, self.D)) * 10.0 if b_prior is None else np.asarray(b_prior, float)

        self.q_nk    = _bgm_init(self.X, self.K, random_state)
        self.alpha_q = self.alpha_prior + np.sum(self.q_nk, axis=0)
        self.a_q     = self.a_prior + self.q_nk.T @ self.X
        self.b_q     = self.b_prior + self.q_nk.T @ (1.0 - self.X)

        self.pi_inferred_mean = self.alpha_q / np.sum(self.alpha_q)
        self.mu_inferred_mean = self.a_q / (self.a_q + self.b_q)
        self.elbo_values      = []
        self.cluster_assignments = None

    def update_q_nk(self):
        log_pi  = digamma(self.alpha_q) - digamma(np.sum(self.alpha_q))
        log_mu  = digamma(self.a_q)     - digamma(self.a_q + self.b_q)
        log_1mu = digamma(self.b_q)     - digamma(self.a_q + self.b_q)
        log_q   = log_pi + np.sum(
            self.X[:, None, :] * log_mu[None, :, :]
            + (1.0 - self.X[:, None, :]) * log_1mu[None, :, :], axis=2)
        log_q  -= logsumexp(log_q, axis=1, keepdims=True)
        self.q_nk = np.exp(log_q)

    def update_pi(self):
        self.alpha_q = self.alpha_prior + np.sum(self.q_nk, axis=0)
        self.pi_inferred_mean = self.alpha_q / np.sum(self.alpha_q)

    def update_mu(self):
        self.a_q = self.a_prior + self.q_nk.T @ self.X
        self.b_q = self.b_prior + self.q_nk.T @ (1.0 - self.X)
        self.mu_inferred_mean = self.a_q / (self.a_q + self.b_q)

    def compute_elbo(self):
        log_mu  = digamma(self.a_q)     - digamma(self.a_q + self.b_q)
        log_1mu = digamma(self.b_q)     - digamma(self.a_q + self.b_q)
        log_pi  = digamma(self.alpha_q) - digamma(np.sum(self.alpha_q))

        ll      = np.sum(self.q_nk[:, :, None] * (
                      self.X[:, None, :] * log_mu[None, :, :]
                    + (1.0 - self.X[:, None, :]) * log_1mu[None, :, :]))
        lz      = np.sum(self.q_nk * log_pi)
        lp_pi   = np.sum((self.alpha_prior - 1.0) * log_pi)
        lp_mu   = np.sum((self.a_prior - 1.0) * log_mu + (self.b_prior - 1.0) * log_1mu)
        h_z     = -np.sum(self.q_nk * np.log(self.q_nk + 1e-10))
        alpha_0 = np.sum(self.alpha_q)
        h_dir   = (gammaln(alpha_0) - np.sum(gammaln(self.alpha_q))
                   + (alpha_0 - self.K) * digamma(alpha_0)
                   - np.sum((self.alpha_q - 1.0) * digamma(self.alpha_q)))
        h_beta  = np.sum(
            gammaln(self.a_q + self.b_q) - gammaln(self.a_q) - gammaln(self.b_q)
            - (self.a_q - 1.0) * digamma(self.a_q)
            - (self.b_q - 1.0) * digamma(self.b_q)
            + (self.a_q + self.b_q - 2.0) * digamma(self.a_q + self.b_q))
        return float(ll + lz + lp_pi + lp_mu + h_z + h_dir + h_beta)

    def fit(self, verbose_every=10):
        score_old = -np.inf
        self.elbo_values = []
        for it in range(self.max_iter):
            self.update_q_nk()
            self.update_pi()
            self.update_mu()
            score = self.compute_elbo()
            self.elbo_values.append(score)
            if abs(score - score_old) < self.tol:
                if verbose_every:
                    print(f"[VIRSE_VI] Converged at iter {it}.")
                break
            score_old = score
            if verbose_every and it % verbose_every == 0:
                print(f"[VIRSE_VI] iter={it:4d}  score={score:.4f}")
        self.cluster_assignments = np.argmax(self.q_nk, axis=1)
        return self

    def predict(self):
        if self.cluster_assignments is None:
            raise ValueError("Run fit() first.")
        return self.cluster_assignments

    def get_results(self):
        return {
            "pi_inferred_mean":    self.pi_inferred_mean,
            "mu_inferred_mean":    self.mu_inferred_mean,
            "cluster_assignments": self.cluster_assignments,
            "q_nk":                self.q_nk,
            "elbo_values":         self.elbo_values,
        }

    def save_results(self, out_dir='.'):
        import os; os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame([self.pi_inferred_mean],
                     columns=[f'K{k}' for k in range(self.K)]).to_csv(
            os.path.join(out_dir, 'VIRSE_VI_pi.csv'), index=False)
        pd.DataFrame(self.mu_inferred_mean).to_csv(
            os.path.join(out_dir, 'VIRSE_VI_mu.csv'), index=False)
        pd.DataFrame({'cluster': self.cluster_assignments}).to_csv(
            os.path.join(out_dir, 'VIRSE_VI_assignments.csv'), index=False)


# Backward-compat alias
VARISEM = VIRSE_VI


# ===========================================================================
#  Gibbs — blocked Gibbs sampler
# ===========================================================================

class Gibbs:
    """
    Blocked Gibbs sampler for the Beta-Bernoulli mixture model.

    Samples π, μ, and z iteratively from their full conditionals.
    Provides posterior samples and posterior-mean estimates.
    """

    def __init__(self, X, K, n_iter=4000, burn_in=1000, thin=5,
                 alpha_prior=None, a_prior=None, b_prior=None,
                 random_state=42):
        self.X = np.asarray(X, dtype=int)
        self.N, self.D = self.X.shape
        self.K = int(K)
        self.n_iter  = int(n_iter)
        self.burn_in = int(burn_in)
        self.thin    = int(thin)
        self.rng     = np.random.RandomState(random_state)

        self.alpha_prior = np.ones(self.K) * 10.0  if alpha_prior is None else np.asarray(alpha_prior, float)
        self.a_prior     = np.ones((self.K, self.D)) * 5.0  if a_prior is None else np.asarray(a_prior, float)
        self.b_prior     = np.ones((self.K, self.D)) * 10.0 if b_prior is None else np.asarray(b_prior, float)

        self.z  = self.rng.choice(self.K, size=self.N)
        self.pi = np.ones(self.K) / self.K
        self.mu = self.rng.beta(self.a_prior, self.b_prior)

        self.pi_samples = []
        self.mu_samples = []
        self.z_samples  = []

        self.pi_inferred_mean    = None
        self.mu_inferred_mean    = None
        self.cluster_assignments = None

    def _sample_pi(self):
        counts = np.bincount(self.z, minlength=self.K)
        self.pi = self.rng.dirichlet(self.alpha_prior + counts)

    def _sample_mu(self):
        mu_new = np.zeros((self.K, self.D))
        for k in range(self.K):
            idx = (self.z == k)
            if np.any(idx):
                success = self.X[idx].sum(axis=0)
                failure = idx.sum() - success
            else:
                success = np.zeros(self.D)
                failure = np.zeros(self.D)
            mu_new[k] = self.rng.beta(self.a_prior[k] + success,
                                      self.b_prior[k] + failure)
        self.mu = np.clip(mu_new, 1e-10, 1 - 1e-10)

    def _sample_z(self):
        log_pi  = np.log(np.clip(self.pi,       1e-12, 1.0))
        log_mu  = np.log(np.clip(self.mu,        1e-12, 1 - 1e-12))
        log_1mu = np.log(np.clip(1.0 - self.mu, 1e-12, 1.0))
        for n in range(self.N):
            lp = log_pi + np.sum(
                self.X[n][None, :] * log_mu
                + (1 - self.X[n])[None, :] * log_1mu, axis=1)
            lp -= logsumexp(lp)
            self.z[n] = self.rng.choice(self.K, p=np.exp(lp))

    def fit(self, verbose_every=500):
        self.pi_samples, self.mu_samples, self.z_samples = [], [], []
        for it in range(self.n_iter):
            self._sample_pi()
            self._sample_mu()
            self._sample_z()
            if it >= self.burn_in and (it - self.burn_in) % self.thin == 0:
                self.pi_samples.append(self.pi.copy())
                self.mu_samples.append(self.mu.copy())
                self.z_samples.append(self.z.copy())
            if verbose_every and (it % verbose_every == 0 or it == self.n_iter - 1):
                counts = np.bincount(self.z, minlength=self.K)
                print(f"[Gibbs] iter={it}  counts={counts}")

        self.pi_samples = np.array(self.pi_samples)
        self.mu_samples = np.array(self.mu_samples)
        self.z_samples  = np.array(self.z_samples)

        self.pi_inferred_mean = self.pi_samples.mean(axis=0)
        self.mu_inferred_mean = self.mu_samples.mean(axis=0)

        z_mode = np.zeros(self.N, dtype=int)
        for n in range(self.N):
            z_mode[n] = np.argmax(np.bincount(self.z_samples[:, n], minlength=self.K))
        self.cluster_assignments = z_mode
        return self

    def predict(self):
        if self.cluster_assignments is None:
            raise ValueError("Run fit() first.")
        return self.cluster_assignments

    def get_results(self):
        if self.pi_inferred_mean is None:
            raise ValueError("Run fit() first.")
        return {
            "pi_inferred_mean":    self.pi_inferred_mean,
            "mu_inferred_mean":    self.mu_inferred_mean,
            "cluster_assignments": self.cluster_assignments,
            "pi_samples":          self.pi_samples,
            "mu_samples":          self.mu_samples,
            "z_samples":           self.z_samples,
        }


# Backward-compat alias
BetaBernoulliGibbs = Gibbs
