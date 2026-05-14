#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VARISEM — Variational Bayesian Bernoulli Mixture Models
--------------------------------------------------------
Two inference algorithms are provided:

  VIRSE       — Mean-approx VB (the VIRSE model).
                q_nk uses posterior means pi_k_mean, mu_ki_mean directly.
                Surrogate objective (not full ELBO).

  VARISEM_VI  — Standard conjugate VB-EM.
                q_nk uses E[log pi_k] and E[log mu_ki] (digamma expectations).
                Optimises the true ELBO.

Priors (defaults)
-----------------
  pi   ~ Dirichlet(alpha_prior)   alpha_prior = 10  per component
  mu_k ~ Beta(a_prior, b_prior)   a_prior = 5, b_prior = 10  per feature

Convenience
-----------
  run_varisem(X, K, method='virse'|'vi', **kwargs)  — one-liner fit + results

Backward-compat alias
---------------------
  VARISEM = VIRSE   (for existing code that imports VARISEM)
"""

import numpy as np
from scipy.special import digamma, logsumexp, gammaln
from sklearn.mixture import BayesianGaussianMixture


# ===========================================================================
#  Shared warm-start helper
# ===========================================================================

def _bgm_init(X, K, random_state):
    """Return BayesianGaussianMixture responsibilities for warm-starting."""
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
#  VIRSE — mean-approx variational inference
# ===========================================================================

class VIRSE:
    """
    VIRSE: mean-based variational approximation for a Bayesian Bernoulli mixture.

    q_nk update:
        q_nk ∝ pi_k · Π_i  mu_ki^x_ni · (1-mu_ki)^(1-x_ni)

    where pi_k and mu_ki are posterior means, not E[log·] expectations.
    This corresponds to the VIRSE model described in the manuscript.
    """

    def __init__(self, X, K, max_iter=200, tol=1e-6,
                 alpha_prior=None, a_prior=None, b_prior=None,
                 random_state=42):
        self.X = np.asarray(X, dtype=float)
        self.N, self.D = self.X.shape
        self.K = int(K)
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.RandomState(random_state)

        # Priors
        self.alpha_prior = (np.ones(self.K) * 10.0
                            if alpha_prior is None
                            else np.asarray(alpha_prior, float))
        self.a_prior = (np.ones((self.K, self.D)) * 5.0
                        if a_prior is None
                        else np.asarray(a_prior, float))
        self.b_prior = (np.ones((self.K, self.D)) * 10.0
                        if b_prior is None
                        else np.asarray(b_prior, float))

        # Warm-start responsibilities via BayesianGaussianMixture
        self.q_nk = _bgm_init(self.X, self.K, random_state)  # (N, K)

        # Posterior parameters
        self.alpha_q = self.alpha_prior + np.sum(self.q_nk, axis=0)
        self.a_q = self.a_prior + self.q_nk.T @ self.X
        self.b_q = self.b_prior + self.q_nk.T @ (1.0 - self.X)

        # Posterior means
        self.pi_k_mean = self.alpha_q / np.sum(self.alpha_q)
        self.mu_ki_mean = self.a_q / (self.a_q + self.b_q)

        self.elbo_values = []
        self.cluster_assignments = None

    # ------------------------------------------------------------------
    #  Update steps
    # ------------------------------------------------------------------

    def update_q_nk(self):
        """
        log q_nk ∝ log pi_k + Σ_i [x_ni log mu_ki + (1-x_ni) log(1-mu_ki)]
        """
        eps = 1e-12
        pi = np.clip(self.pi_k_mean, eps, 1.0 - eps)    # (K,)
        mu = np.clip(self.mu_ki_mean, eps, 1.0 - eps)   # (K, D)

        log_q = (np.log(pi)[None, :]
                 + np.einsum("nd,kd->nk", self.X, np.log(mu))
                 + np.einsum("nd,kd->nk", 1.0 - self.X, np.log(1.0 - mu)))

        log_q -= logsumexp(log_q, axis=1, keepdims=True)
        self.q_nk = np.exp(log_q)

    def update_pi_k(self):
        """Dirichlet posterior update; store posterior mean."""
        self.alpha_q = self.alpha_prior + np.sum(self.q_nk, axis=0)
        self.pi_k_mean = self.alpha_q / np.sum(self.alpha_q)

    def update_mu_ki(self):
        """Beta posterior update; store posterior mean."""
        self.a_q = self.a_prior + self.q_nk.T @ self.X
        self.b_q = self.b_prior + self.q_nk.T @ (1.0 - self.X)
        self.mu_ki_mean = self.a_q / (self.a_q + self.b_q)

    # ------------------------------------------------------------------
    #  Surrogate objective (mean-approx ELBO)
    # ------------------------------------------------------------------

    def compute_elbo(self):
        """
        Mean-based surrogate objective consistent with the mean-approx update.

        Uses posterior means as plug-ins inside the expected
        complete-data log-likelihood.  Not identical to the standard
        conjugate-VB ELBO, but monotone under these updates.
        """
        eps = 1e-12
        pi = np.clip(self.pi_k_mean, eps, 1.0 - eps)
        mu = np.clip(self.mu_ki_mean, eps, 1.0 - eps)

        log_pi  = np.log(pi)
        log_mu  = np.log(mu)
        log_1mu = np.log(1.0 - mu)

        # E[log p(X | Z, μ)]
        ll = np.sum(
            self.q_nk[:, :, None] * (
                self.X[:, None, :] * log_mu[None, :, :]
                + (1.0 - self.X[:, None, :]) * log_1mu[None, :, :]
            )
        )

        # E[log p(Z | π)]
        lz = np.sum(self.q_nk * log_pi)

        # log p(π) — Dirichlet prior (up to constant)
        lp_pi = np.sum((self.alpha_prior - 1.0) * log_pi)

        # log p(μ) — Beta prior (up to constant)
        lp_mu = np.sum(
            (self.a_prior - 1.0) * log_mu
            + (self.b_prior - 1.0) * log_1mu
        )

        # H[q(Z)]
        h_z = -np.sum(self.q_nk * np.log(self.q_nk + eps))

        return float(ll + lz + lp_pi + lp_mu + h_z)

    # ------------------------------------------------------------------
    #  Fit
    # ------------------------------------------------------------------

    def fit(self, verbose_every=10):
        """Run coordinate-ascent updates until convergence or max_iter."""
        score_old = -np.inf
        self.elbo_values = []

        for it in range(self.max_iter):
            self.update_q_nk()
            self.update_pi_k()
            self.update_mu_ki()

            score = self.compute_elbo()
            self.elbo_values.append(score)

            if abs(score - score_old) < self.tol:
                print(f"Converged at iteration {it}.")
                break

            score_old = score

            if it % verbose_every == 0:
                print(f"Iter {it:4d} | score = {score:.4f}")

        self.cluster_assignments = np.argmax(self.q_nk, axis=1)
        print(f"Done.  π̂ = {np.round(self.pi_k_mean, 3)}")

    # ------------------------------------------------------------------
    #  Results
    # ------------------------------------------------------------------

    def get_results(self):
        return {
            "pi_inferred_mean":    self.pi_k_mean,
            "mu_inferred_mean":    self.mu_ki_mean,
            "cluster_assignments": self.cluster_assignments,
            "q_nk":                self.q_nk,
            "elbo_values":         self.elbo_values,
        }


# ===========================================================================
#  VARISEM_VI — standard conjugate VB-EM (digamma / full ELBO)
# ===========================================================================

class VARISEM_VI:
    """
    Standard conjugate variational Bayes for a Bayesian Bernoulli mixture.

    q_nk update uses E[log pi_k] and E[log mu_ki] (digamma expectations),
    maximising the true ELBO (evidence lower bound).
    """

    def __init__(self, X, K, max_iter=200, tol=1e-6,
                 alpha_prior=None, a_prior=None, b_prior=None,
                 random_state=42):
        self.X = np.asarray(X, dtype=float)
        self.N, self.D = self.X.shape
        self.K = int(K)
        self.max_iter = max_iter
        self.tol = tol
        self.rng = np.random.RandomState(random_state)

        # Priors
        self.alpha_prior = (np.ones(self.K) * 10.0
                            if alpha_prior is None
                            else np.asarray(alpha_prior, float))
        self.a_prior = (np.ones((self.K, self.D)) * 5.0
                        if a_prior is None
                        else np.asarray(a_prior, float))
        self.b_prior = (np.ones((self.K, self.D)) * 10.0
                        if b_prior is None
                        else np.asarray(b_prior, float))

        # Warm-start via BayesianGaussianMixture
        self.q_nk = _bgm_init(self.X, self.K, random_state)  # (N, K)

        # Posterior parameters
        self.alpha_q = self.alpha_prior + np.sum(self.q_nk, axis=0)
        self.a_q = self.a_prior + self.q_nk.T @ self.X
        self.b_q = self.b_prior + self.q_nk.T @ (1.0 - self.X)

        # Posterior means (convenience attributes)
        self.pi_k_mean = self.alpha_q / np.sum(self.alpha_q)
        self.mu_ki_mean = self.a_q / (self.a_q + self.b_q)

        self.elbo_values = []
        self.cluster_assignments = None

    # ------------------------------------------------------------------
    #  Update steps
    # ------------------------------------------------------------------

    def update_q_nk(self):
        """
        Standard VB-EM E-step using digamma expectations:
            log q_nk ∝ E[log pi_k] + Σ_i E[log mu_ki]·x_ni + E[log(1-mu_ki)]·(1-x_ni)
        """
        log_pi_k   = digamma(self.alpha_q) - digamma(np.sum(self.alpha_q))
        log_mu_ki  = digamma(self.a_q) - digamma(self.a_q + self.b_q)
        log_1mu_ki = digamma(self.b_q) - digamma(self.a_q + self.b_q)

        log_q = log_pi_k + np.sum(
            self.X[:, None, :] * log_mu_ki[None, :, :]
            + (1.0 - self.X[:, None, :]) * log_1mu_ki[None, :, :],
            axis=2,
        )
        log_q -= logsumexp(log_q, axis=1, keepdims=True)
        self.q_nk = np.exp(log_q)

    def update_pi_k(self):
        """Dirichlet posterior update; refresh posterior mean."""
        self.alpha_q  = self.alpha_prior + np.sum(self.q_nk, axis=0)
        self.pi_k_mean = self.alpha_q / np.sum(self.alpha_q)

    def update_mu_ki(self):
        """Beta posterior update; refresh posterior mean."""
        self.a_q = self.a_prior + self.q_nk.T @ self.X
        self.b_q = self.b_prior + self.q_nk.T @ (1.0 - self.X)
        self.mu_ki_mean = self.a_q / (self.a_q + self.b_q)

    # ------------------------------------------------------------------
    #  True ELBO
    # ------------------------------------------------------------------

    def compute_elbo(self):
        """
        Full conjugate-VB ELBO:
          E[log p(X|Z,μ)] + E[log p(Z|π)] + E[log p(π)] + E[log p(μ)]
          + H[q(Z)] + H[q(π)] + H[q(μ)]
        """
        log_mu_ki  = digamma(self.a_q) - digamma(self.a_q + self.b_q)
        log_1mu_ki = digamma(self.b_q) - digamma(self.a_q + self.b_q)
        log_pi_k   = digamma(self.alpha_q) - digamma(np.sum(self.alpha_q))

        # E[log p(X | Z, μ)]
        log_likelihood = np.sum(
            self.q_nk[:, :, None] * (
                self.X[:, None, :] * log_mu_ki[None, :, :]
                + (1.0 - self.X[:, None, :]) * log_1mu_ki[None, :, :]
            )
        )

        # E[log p(Z | π)]
        log_prior_z  = np.sum(self.q_nk * log_pi_k)
        # E[log p(π | α_0)]
        log_prior_pi = np.sum((self.alpha_prior - 1.0) * log_pi_k)
        # E[log p(μ | a, b)]
        log_prior_mu = np.sum(
            (self.a_prior - 1.0) * log_mu_ki
            + (self.b_prior - 1.0) * log_1mu_ki
        )

        # H[q(Z)]
        entropy_q_z = -np.sum(self.q_nk * np.log(self.q_nk + 1e-10))

        # H[q(π)] — Dirichlet entropy
        alpha_0 = np.sum(self.alpha_q)
        dirichlet_entropy = (
            gammaln(alpha_0) - np.sum(gammaln(self.alpha_q))
            + (alpha_0 - self.K) * digamma(alpha_0)
            - np.sum((self.alpha_q - 1.0) * digamma(self.alpha_q))
        )

        # H[q(μ)] — Beta entropy (sum over all k, i)
        beta_entropy = np.sum(
            gammaln(self.a_q + self.b_q) - gammaln(self.a_q) - gammaln(self.b_q)
            - (self.a_q - 1.0) * digamma(self.a_q)
            - (self.b_q - 1.0) * digamma(self.b_q)
            + (self.a_q + self.b_q - 2.0) * digamma(self.a_q + self.b_q)
        )

        return float(log_likelihood + log_prior_z + log_prior_pi + log_prior_mu
                     + entropy_q_z + dirichlet_entropy + beta_entropy)

    # ------------------------------------------------------------------
    #  Fit
    # ------------------------------------------------------------------

    def fit(self, verbose_every=10):
        """Run VB-EM coordinate-ascent until convergence or max_iter."""
        elbo_old = -np.inf
        self.elbo_values = []

        for it in range(self.max_iter):
            self.update_q_nk()
            self.update_pi_k()
            self.update_mu_ki()

            elbo = self.compute_elbo()
            self.elbo_values.append(elbo)

            if abs(elbo - elbo_old) < self.tol:
                print(f"Converged at iteration {it}.")
                break
            elbo_old = elbo

            if it % verbose_every == 0:
                print(f"Iter {it:4d} | ELBO = {elbo:.4f}")

        self.cluster_assignments = np.argmax(self.q_nk, axis=1)
        print(f"Done.  π̂ = {np.round(self.pi_k_mean, 3)}")

    # ------------------------------------------------------------------
    #  Results
    # ------------------------------------------------------------------

    def get_results(self):
        return {
            "pi_inferred_mean":    self.pi_k_mean,
            "mu_inferred_mean":    self.mu_ki_mean,
            "cluster_assignments": self.cluster_assignments,
            "q_nk":                self.q_nk,
            "elbo_values":         self.elbo_values,
        }


# ===========================================================================
#  Backward-compat alias  (existing code: from VARISEM import VARISEM)
# ===========================================================================

VARISEM = VIRSE


# ===========================================================================
#  Convenience runner
# ===========================================================================

def run_varisem(X, K, method="virse", **kwargs):
    """
    Fit a Bayesian Bernoulli mixture and return the results dict.

    Parameters
    ----------
    X       : array-like (N, D)  — binary data matrix
    K       : int                — number of mixture components
    method  : str                — 'virse' (default) or 'vi'
                'virse'  → VIRSE        (mean-approx, surrogate objective)
                'vi'     → VARISEM_VI   (digamma E[log·], true ELBO)
    **kwargs: passed to the chosen class  (max_iter, tol, random_state, …)

    Returns
    -------
    dict with keys: pi_inferred_mean, mu_inferred_mean,
                    cluster_assignments, q_nk, elbo_values
    """
    method = method.lower()
    if method in ("virse", "mean", "approx"):
        model = VIRSE(X, K, **kwargs)
    elif method in ("vi", "vb", "varisem_vi"):
        model = VARISEM_VI(X, K, **kwargs)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'virse' or 'vi'.")

    model.fit()
    return model.get_results()
