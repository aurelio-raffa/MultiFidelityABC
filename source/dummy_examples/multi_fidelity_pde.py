import pde

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from time import time
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from source.models.high_fidelity import HighFidelityModel
from source.models.low_fidelity import LowFidelityModel
from source.core.metropolis_hastings import metropolis_hastings, adaptive_multifidelity_mh

if __name__ == '__main__':
    np.random.seed(1251)

    class DiffPDE:
        def __init__(self, gridscale_x, gridscale_y, init_state_expression, t):
            self.grid = pde.UnitGrid([gridscale_x, gridscale_y])
            self.scale = np.array([gridscale_x, gridscale_y])
            self.t = t
            self.init_state = pde.ScalarField.from_expression(self.grid, init_state_expression)

        def plot_init_state(self):
            self.init_state.plot()

        def eval(self, z):
            eq = pde.DiffusionPDE(diffusivity=np.exp(z))
            result = eq.solve(self.init_state, t_range=self.t)
            return result

        def __call__(self, z, x):
            nd = x.shape[1]
            if type(z) in [int, float, np.int, np.float, np.float32, np.float64]:
                n_evals = 1
                z = np.array([z])
            else:
                n_evals = len(z)
            return_mat = np.zeros((n_evals, nd))
            for i, z_ in enumerate(z):
                result = self.eval(z_)
                for j in range(x.shape[1]):
                    return_mat[i, j] = result.interpolate(x[:, j])
            return return_mat

    forward_model = DiffPDE(64, 64, 'sin(x/5) * cos(y/5) * exp(-((x-50)/30)**2 -((y-50)/30)**2) * 100', 50)

    np.random.seed(1251)

    xs = np.random.uniform(0, 1, size=(2, 20))

    true_z = -0.25
    noise_sigma = .2
    prior_mean = 0
    prior_sigma = 1
    num_data = 50
    num_quad = 10
    poly_order = 3
    multi_q = 5
    x_min = 0
    x_max = 1

    def err_dens(x_):
        rv = norm(0, noise_sigma)
        return np.prod([rv.pdf(el) for el in x_])


    def prior(x_):
        rv = norm(prior_mean, prior_sigma)
        if type(x_) is np.array:
            return np.prod(np.array([rv.pdf(el) for el in x_]))
        else:
            return rv.pdf(x_)


    x = np.random.uniform(0, 1, size=(2, num_data))
    y = forward_model(true_z, x)
    data = y + np.random.normal(0, noise_sigma, len(y))

    hfm = HighFidelityModel(
        forward_model, data, x, err_dens, prior)

    lfm = LowFidelityModel(
        data, err_dens, prior, poly_order, multi_q)

    zs = np.random.normal(prior_mean, prior_sigma, num_quad)
    evals = forward_model(zs, x)
    lfm.fit(zs, evals)

    class UnifProposal:
        def __init__(self, r):
            self.r = r

        def draw(self, z):
            return z + np.random.uniform(-self.r, self.r)

        def density(self, z1, z2):
            return 1. / (2 * self.r) if np.abs(z1 - z2) <= self.r else 0

    proposal = UnifProposal(.1)

    samples = 500
    init_z = 0.
    subchain_len = 25
    upper_thr = 1e-2
    error_thr = 1e-4
    init_radius = .5
    rho_factor = .8

    mfm = deepcopy(lfm)

    t_ = time()
    lfmh = metropolis_hastings(lfm, proposal, init_z, samples).flatten()
    print('low-fidelity:\t', time() - t_)
    t_ = time()
    mfmh = adaptive_multifidelity_mh(
        subchain_len,
        samples // subchain_len,
        upper_thr,
        error_thr,
        init_radius,
        rho_factor,
        mfm,
        hfm,
        proposal,
        init_z).flatten()
    print('multi-fidelity:\t', time() - t_)
    t_ = time()
    hfmh = metropolis_hastings(hfm, proposal, init_z, samples).flatten()
    print('high-fidelity:\t', time() - t_)

    plt.figure()

    plt.subplot(1, 2, 1)
    plt.plot(hfmh, label='true model MH')
    plt.plot(lfmh, label='surrogate model MH')
    plt.plot(mfmh, label='adaptive MH')
    plt.legend()

    burn = 0
    plt.subplot(1, 2, 2)
    plt.title('models evaluated at true parameter')
    mh_data = pd.DataFrame({
        'data': np.concatenate([hfmh[burn:], lfmh[burn:], mfmh[burn:]], axis=0),
        'method':
            ['true model MH samples'] * (samples - burn) +
            ['surrogate model MH samples'] * (samples - burn) +
            ['adaptive MH samples'] * (samples - burn)})
    sns.histplot(
        mh_data,
        x='data', hue='method',
        multiple='layer',
        edgecolor='.3',
        linewidth=.5)
    plt.show()



