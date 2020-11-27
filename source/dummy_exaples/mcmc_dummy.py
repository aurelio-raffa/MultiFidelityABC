import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from source.models.high_fidelity import HighFidelityModel
from source.core.metropolis_hastings import metropolis_hastings

if __name__ == '__main__':
    def cfun(z, x_):
        return np.power(x_, z)

    true_k = 5
    noise_sigma = .1
    prior_mean = 0
    prior_sigma = 5
    num_data = 20

    x = np.linspace(0, 1, num_data)
    y = cfun(true_k, x)
    data = y + np.random.normal(0, noise_sigma, len(y))

    plt.figure()
    plt.plot(x, y)
    plt.plot(x, data)
    plt.show()

    def err_dens(x_):
        rv = norm(0, noise_sigma)
        return np.prod([rv.pdf(el) for el in x_])

    def prior(x_):
        rv = norm(prior_mean, prior_sigma)
        if type(x_) is np.array:
            return np.prod(np.array([rv.pdf(el) for el in x_]))
        else:
            return rv.pdf(x_)

    hfm = HighFidelityModel(
        cfun,
        data,
        x,
        err_dens,
        prior)

    class UnifProposal:
        def __init__(self, r):
            self.r = r

        def draw(self, z):
            return z + np.random.uniform(-self.r, self.r)

        def density(self, z1, z2):
            return 1./(2 * self.r) if np.abs(z1 - z2) <= self.r else 0

    proposal = UnifProposal(1)

    res = metropolis_hastings(hfm, proposal, 0., 1000).flatten()
    plt.figure()
    plt.plot(range(len(res)), res)
    plt.show()

