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


class IndComponentsGaussian:
    def __init__(self, sigmas):
        self.sigmas = sigmas
        self._distr = [GaussDensity(sig) for sig in sigmas]

    def __call__(self, x_):
        assert x_.size == len(self._distr)
        return np.sum(np.array([dis(xx) for dis, xx in zip(self._distr, x_.flatten())]))

    def draw(self, z):
        assert z.size == len(self._distr)
        return np.array([dis.draw(z_) for dis, z_ in zip(self._distr, z.flatten())]).reshape(z.shape)

    def logdensity(self, z1, z2):
        assert z1.size == z2.size == len(self._distr)
        return np.sum(np.array([dis(z1_ - z2_) for dis, z1_, z2_ in zip(self._distr, z1.flatten(), z2.flatten())]))

