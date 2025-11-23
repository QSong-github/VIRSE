import numpy as np
import pandas as pd
from scipy.special import digamma, logsumexp, gammaln
from sklearn.mixture import BayesianGaussianMixture

class VARISEM:
    def __init__(self, X, K, max_iter=300, tol=1e-6, alpha_prior=None, a_prior=None, b_prior=None):
        self.X = X
        self.N, self.D = X.shape
        self.K = K
        self.max_iter = max_iter
        self.tol = tol

        # Priors
        self.alpha_prior = alpha_prior if alpha_prior is not None else np.ones(K) * 1
        self.a_prior = a_prior if a_prior is not None else np.ones((K, self.D)) * 1
        self.b_prior = b_prior if b_prior is not None else np.ones((K, self.D)) * 1

        # Initialize q_nk with BayesianGaussianMixture
        bmm = BayesianGaussianMixture(
            n_components=K,
            covariance_type="full",
            weight_concentration_prior_type="dirichlet_process",
            random_state=42,
            max_iter=100,
            n_init=10
        )
        bmm.fit(X)
        self.q_nk = bmm.predict_proba(X)

        # Initialize variational parameters
        self.alpha_q = self.alpha_prior + np.sum(self.q_nk, axis=0)
        self.a_q = self.a_prior + np.dot(self.q_nk.T, X)
        self.b_q = self.b_prior + np.dot(self.q_nk.T, 1 - X)

        self.pi_inferred_mean = None
        self.mu_inferred_mean = None
        self.cluster_assignments = None

        self.elbo_values = []

    def update_q_nk(self):
        log_pi_k = digamma(self.alpha_q) - digamma(np.sum(self.alpha_q))
        log_mu_ki = digamma(self.a_q) - digamma(self.a_q + self.b_q)
        log_1_mu_ki = digamma(self.b_q) - digamma(self.a_q + self.b_q)

        log_q_nk = log_pi_k + np.sum(
            self.X[:, None, :] * log_mu_ki[None, :, :] +
            (1 - self.X[:, None, :]) * log_1_mu_ki[None, :, :],
            axis=2
        )
        log_q_nk -= logsumexp(log_q_nk, axis=1, keepdims=True)
        self.q_nk = np.exp(log_q_nk)

    def update_pi_k(self):
        self.alpha_q = self.alpha_prior + np.sum(self.q_nk, axis=0)
        self.pi_inferred_mean = self.alpha_q / np.sum(self.alpha_q)

    def update_mu_ki(self):
        self.a_q = self.a_prior + np.dot(self.q_nk.T, self.X)
        self.b_q = self.b_prior + np.dot(self.q_nk.T, 1 - self.X)
        self.mu_inferred_mean = self.a_q / (self.a_q + self.b_q)

    def compute_elbo(self):
        log_mu_ki = digamma(self.a_q) - digamma(self.a_q + self.b_q)
        log_1_mu_ki = digamma(self.b_q) - digamma(self.a_q + self.b_q)

        log_likelihood = np.sum(self.q_nk[:, :, None] * (
            self.X[:, None, :] * log_mu_ki[None, :, :] +
            (1 - self.X[:, None, :]) * log_1_mu_ki[None, :, :]
        ))

        log_pi_k = digamma(self.alpha_q) - digamma(np.sum(self.alpha_q))
        log_prior_z = np.sum(self.q_nk * log_pi_k)
        log_prior_pi = np.sum((self.alpha_prior - 1) * log_pi_k)
        log_prior_mu = np.sum((self.a_prior - 1) * log_mu_ki + (self.b_prior - 1) * log_1_mu_ki)

        entropy_q_z = -np.sum(self.q_nk * np.log(self.q_nk + 1e-10))
        alpha_0 = np.sum(self.alpha_q)
        dirichlet_entropy = (
            gammaln(alpha_0) - np.sum(gammaln(self.alpha_q)) +
            (alpha_0 - self.K) * digamma(alpha_0) -
            np.sum((self.alpha_q - 1) * digamma(self.alpha_q))
        )
        beta_entropy = np.sum(
            gammaln(self.a_q + self.b_q) -
            gammaln(self.a_q) -
            gammaln(self.b_q) -
            (self.a_q - 1) * digamma(self.a_q) -
            (self.b_q - 1) * digamma(self.b_q) +
            (self.a_q + self.b_q - 2) * digamma(self.a_q + self.b_q)
        )

        elbo = (
            log_likelihood +
            log_prior_z +
            log_prior_pi +
            log_prior_mu +
            entropy_q_z +
            dirichlet_entropy +
            beta_entropy
        )
        return elbo

    def fit(self):
        elbo_old = -np.inf
        self.elbo_values = []

        for iteration in range(self.max_iter):
            self.update_q_nk()
            self.update_pi_k()
            self.update_mu_ki()

            elbo = self.compute_elbo()
            self.elbo_values.append(elbo)

            if np.abs(elbo - elbo_old) < self.tol:
                print(f"Converged at iteration {iteration}, ELBO: {elbo:.4f}")
                break

            elbo_old = elbo

            if iteration % 10 == 0 or iteration == self.max_iter - 1:
                print(f"Iteration {iteration}, ELBO: {elbo:.4f}")

        self.cluster_assignments = np.argmax(self.q_nk, axis=1)

    def predict(self):
        if self.cluster_assignments is None:
            raise ValueError("Model not yet fitted. Run .fit() first.")
        return self.cluster_assignments

    def get_results(self):
        if self.pi_inferred_mean is None or self.mu_inferred_mean is None:
            raise ValueError("Model not yet fitted. Run .fit() first.")

        return {
            "pi_inferred_mean": self.pi_inferred_mean,
            "mu_inferred_mean": self.mu_inferred_mean,
            "cluster_assignments": self.cluster_assignments,
            "elbo_values": self.elbo_values
        }

    def save_results(self, pi_filename="pi_inferred_mean.csv",
                     mu_filename="mu_inferred_mean.csv",
                     cluster_filename="VARISEM_cluster_assignments.csv"):
        """
        Save inferred parameters and cluster assignments to separate CSV files.

        Args:
            pi_filename (str): CSV file path to save pi (cluster proportions).
            mu_filename (str): CSV file path to save mu (mutation probabilities).
            cluster_filename (str): CSV file path to save predicted cluster assignments.
        """
        if self.pi_inferred_mean is None or self.mu_inferred_mean is None or self.cluster_assignments is None:
            raise ValueError("Model not yet fitted. Run .fit() first.")

        # Save pi (cluster proportions)
        pi_df = pd.DataFrame([self.pi_inferred_mean], columns=[f"Cluster_{k}" for k in range(self.K)])
        pi_df.to_csv(pi_filename, index=False)

        # Save mu (mutation probabilities for each cluster)
        mu_df = pd.DataFrame(self.mu_inferred_mean, columns=[f"Feature_{d}" for d in range(self.D)])
        mu_df.index = [f"Cluster_{k}" for k in range(self.K)]
        mu_df.to_csv(mu_filename)

        # Save cluster assignments
        cluster_df = pd.DataFrame({
            "Sample_ID": np.arange(self.N),
            "Cluster_Assignment": self.cluster_assignments
        })
        cluster_df.to_csv(cluster_filename, index=False)

        print(f"Results saved to:\n- pi: {pi_filename}\n- mu: {mu_filename}\n- Clusters: {cluster_filename}")
