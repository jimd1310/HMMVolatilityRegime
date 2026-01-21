"""
Hidden Markov Model for modelling Volatility Regimes.

This module implements a Gaussian HMM on 100 x log(returns).
"""
from .base import Base
import numpy as np
from scipy.stats import norm
from scipy.special import logsumexp
from hmmlearn import hmm


class GHMM(Base):
    """
    Creates an instance of the Gaussian HMM for volatility regime detection.

    Parameters
    ---------------------
    n_states : int, default=2
        Number of hidden states in the HMM.
    
    random_state : int or None, default=None
        Random seed for reproducibility.
    """
    def __init__(self, n_states=2, random_state=None):
        self.n_states = n_states
        self.random_state = random_state
        self.model = None
        self.loglik = None
        self.n_params = (
            self.n_states - 1 + # initial state distribution
            self.n_states * (self.n_states - 1) + # transition matrix
            self.n_states * 2 # means and variances
        )
    

    def fit(self, logret):
        """
        Fits the HMM to log return data.

        Parameters
        ----------
        logret : NumPy array of shape (n_samples, 1)
            Log return time series data in percentage (x100).

        Returns
        -------
        self : GHMM
            The fitted GHMM instance.
        """
        data = np.asarray(logret).reshape(-1, 1)

        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            n_iter=1000,
            covariance_type='diag',
            tol=1e-4,
            random_state=self.random_state
        )
        self.model.fit(data)
        self.loglik = self.model.score(data)
        
        return self
    
    
    def log_predictive_density(self, logret_test, logret): 
        """
        Compute the log predictive density of the test data.

        Parameters
        ----------
        logret_test : NumPy array of shape (n_samples, 1)
            Test log return time series data in percentage (x100).

        Returns
        -------
        log_scores : NumPy array of shape (n_samples, 1)
            Log predictive densities for each test data point.
        """
        data = np.asarray(logret_test).ravel()
        P = self.model.transmat_
        means = self.model.means_.flatten()
        stds = np.sqrt(self.model.covars_.flatten())

        # Start from training probabilities
        gamma_train = self.model.predict_proba(
            np.asarray(logret).reshape(-1, 1)
            )
        gamma = gamma_train[-1]
        log_scores = []

        for y_t in data: 
            # Prediction 
            pi_pred = gamma @ P 

            # Predictive Mixture Density
            log_f = logsumexp(
                np.log(pi_pred) + norm.logpdf(y_t, loc=means, scale=stds)
            )
            log_scores.append(log_f)

            # Filtering Update
            likelihoods = norm.pdf(y_t, loc=means, scale=stds)
            gamma = pi_pred * likelihoods
            gamma /= gamma.sum() # normalise

        return np.array(log_scores)