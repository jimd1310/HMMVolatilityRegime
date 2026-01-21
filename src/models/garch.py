"""
GARCH(1,1) model for time-varying volatility, used for benchmarking.

This module implements a GARCH(1,1) model on 100 x log(returns).
"""
from .base import Base
import numpy as np
from scipy.stats import norm
from arch import arch_model 


class GARCH(Base):
    """
    Creates an instance of GARCH(1,1) model for volatility benchmarking with
    Gaussian innovations.
    """
    def __init__(self):
        self.model = None
        self.res = None
        self.loglik = None
        self.n_params = 4  # omega, alpha, beta


    def fit(self, logret):
        """
        Fits the GARCH(1,1) model to log return data.

        Parameters
        ----------
        logret : NumPy array of shape (n_samples, 1)
            Log return time series data in percentage (x100).
        
        Returns
        -------
        self : GARCH
            The fitted GARCH instance.
        """
        data = np.asarray(logret).ravel()

        self.model = arch_model(
            data, 
            mean='zero', 
            vol='GARCH', 
            p=1, 
            q=1, 
            dist='normal'
            )
        self.res = self.model.fit(disp="off")
        self.loglik = self.res.loglikelihood

        return self
    
    
    def log_predictive_density(self, logret_test):
        """
        Computes the log predictive density of the test data.

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
        
        # Get last in-sample conditional variance
        sigma2_prev = self.res.conditional_volatility[-1] ** 2

        # Parameters
        params = self.res.params
        omega = params['omega']
        alpha = params['alpha[1]']
        beta = params['beta[1]']

        log_scores = []

        for y in data: 
            # One step ahead forecast of variance
            sigma2 = omega + alpha * (y ** 2) + beta * sigma2_prev
            sigma = np.sqrt(sigma2)
            
            log_scores.append(
                norm.logpdf(y, loc=0, scale=sigma)
            )
            sigma2_prev = sigma2

        return np.array(log_scores)
