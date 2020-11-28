import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
from source.models.high_fidelity import HighFidelityModel
from source.models.low_fidelity import LowFidelityModel
from source.core.metropolis_hastings import metropolis_hastings, adaptive_multifidelity_mh

if __name__ == '__main__':
    def cfun(z, x_):
        return 5. / (1 + x ** 2) + z * np.cos(1.5 * x_)

    np.random.seed(1251)

    true_k = 5
    noise_sigma = 3
    prior_mean = 0
    prior_sigma = 10
    num_data = 20
    num_quad = 5

    x = np.linspace(0.01, 9.99, num_data)
    y = cfun(true_k, x)
    data = y + np.random.normal(0, noise_sigma, len(y))

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
        cfun, data, x, err_dens, prior)

    lfm = LowFidelityModel(
        data, err_dens, prior, 3, 5)

    zs = np.random.normal(prior_mean, prior_sigma, num_quad)
    lfm.fit(zs, np.concatenate([hfm.eval(z_).reshape(1, -1) for z_ in zs], axis=0))

    plt.figure()
    plt.plot(x, data, label='generated data')
    plt.plot(x, hfm.eval(true_k), label='true model')
    plt.plot(x, lfm.eval(true_k), label='surrogate model')
    plt.legend()
    plt.show()

    class UnifProposal:
        def __init__(self, r):
            self.r = r

        def draw(self, z):
            return z + np.random.uniform(-self.r, self.r)

        def density(self, z1, z2):
            return 1./(2 * self.r) if np.abs(z1 - z2) <= self.r else 0

    proposal = UnifProposal(1)

    samples = 1000
    init_z = 0.
    subchain_len = 10
    upper_thr = 1e-2
    error_thr = 1e-4
    init_radius = 1.
    rho_factor = .8

    plt.figure()
    plt.plot(metropolis_hastings(hfm, proposal, init_z, samples).flatten(), label='true model MH')
    plt.plot(metropolis_hastings(lfm, proposal, init_z, samples).flatten(), label='surrogate model MH')
    plt.plot(
        adaptive_multifidelity_mh(
            subchain_len,
            samples // subchain_len,
            upper_thr,
            error_thr,
            init_radius,
            rho_factor,
            lfm,
            hfm,
            proposal,
            init_z).flatten(),
        label='adaptive MH')
    plt.legend()
    plt.show()
