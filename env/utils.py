import numpy as np
from scipy.special import expit
from scipy.stats import norm


def sigmoid(x, midpoint=0.0, sharpness=1.0):
    """
    Standard sigmoid curve.
    - x: input value
    - midpoint: the x-value where sigmoid = 0.5
    - sharpness: controls steepness (higher = steeper)
    """
    return expit(sharpness * (x - midpoint))


class GaussianMixture:
    def __init__(self, weights, mus, sds, rng=None):
        """
        Create a Gaussian mixture model.

        Parameters
        ----------
        weights : list or tuple of floats
            Mixture weights, must sum to 1.
        mus : list or tuple of floats
            Means of the Gaussian components.
        sds : list or tuple of floats
            Standard deviations of the components.
        rng : np.random.Generator, optional
            Random generator (default: np.random.default_rng()).
        """
        self.weights = np.array(weights)
        self.mus = np.array(mus)
        self.sds = np.array(sds)
        self.rng = rng or np.random.default_rng()

    def pdf(self, x):
        """Evaluate the PDF of the Gaussian mixture at x."""
        pdf_vals = np.zeros_like(x, dtype=float)
        for w, mu, sd in zip(self.weights, self.mus, self.sds):
            pdf_vals += w * norm.pdf(x, loc=mu, scale=sd)
        return pdf_vals

    def sample(self, n=1):
        """Draw n samples from the Gaussian mixture."""
        choices = self.rng.choice(len(self.weights), size=n, p=self.weights)
        samples = np.array([self.rng.normal(self.mus[i], self.sds[i]) for i in choices])
        return samples
