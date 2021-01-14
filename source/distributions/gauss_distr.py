import numpy as np


class GaussDensity:
    def __init__(self, sigma=1., mu=None):
        self.sigma = sigma
        self.mu = mu

    def __call__(self, x_):
        return np.sum(-x_**2/(2*self.sigma**2)) if self.mu is None else np.sum(-(x_-self.mu)**2/(2*self.sigma**2))

    def draw(self, z):
        return np.random.normal(z if self.mu is None else self.mu, self.sigma, size=z.shape)

    def logdensity(self, z1, z2):
        return np.sum(self(z1 - z2)) if self.mu is None else np.sum(self(z1))


class IndComponentsGaussian:
    def __init__(self, sigmas, mus=None):
        self.sigmas = sigmas
        self.mus = mus
        if mus is None:
            self._distr = [GaussDensity(sig) for sig in sigmas]
        else:
            self._distr = [GaussDensity(sig, mu) for sig, mu in zip(sigmas, mus)]

    def __call__(self, x_):
        assert x_.size == len(self._distr)
        return np.sum(np.array([dis(xx) for dis, xx in zip(self._distr, x_.flatten())]))

    def draw(self, z):
        assert z.size == len(self._distr)
        return np.array([dis.draw(z_) for dis, z_ in zip(self._distr, z.flatten())]).reshape(z.shape)

    def logdensity(self, z1, z2):
        assert z1.size == z2.size == len(self._distr)
        return np.sum(np.array([dis(z1_ - z2_) for dis, z1_, z2_ in zip(self._distr, z1.flatten(), z2.flatten())]))
