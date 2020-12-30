import numpy as np

from scipy.stats import multivariate_normal


class GaussDensity:
    def __init__(self, variance=1):
        self.variance = variance # prima chiamata noise_variance

    def __call__(self, x_):
        return np.sum(-x_ ** 2 / (2 * self.variance))

    def draw(self, z):
        return np.random.normal(z, self.variance, size=z.shape)

    def logdensity(self, z1, z2):
        dens = multivariate_normal(z1, self.variance)
        return np.log(dens.pdf(z2))

