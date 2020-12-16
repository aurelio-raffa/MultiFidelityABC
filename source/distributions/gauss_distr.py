import numpy as np


class GaussDensity:
    def __init__(self, noise_variance=1):
        self.noise_variance = noise_variance

    def __call__(self, x_):
        return np.sum(-x_ ** 2 / (2 * self.noise_variance))
