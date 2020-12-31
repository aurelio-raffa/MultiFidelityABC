import numpy as np


class GaussDensity:
    def __init__(self, sigma=1.):
        self.sigma = sigma

    def __call__(self, x_):
        return np.sum(-x_ ** 2 / (2 * self.sigma ** 2))

    def draw(self, z):
        return np.random.normal(z, self.sigma, size=z.shape)

    def logdensity(self, z1, z2):
        return np.sum(self(z1 - z2))

