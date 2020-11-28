import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import norm
from source.models.high_fidelity import HighFidelityModel
from source.models.low_fidelity import LowFidelityModel
from source.core.metropolis_hastings import metropolis_hastings, adaptive_multifidelity_mh

if __name__ == '__main__':
    np.random.seed(1251)

    true_z = 4
    noise_sigma = .2
    prior_mean = 0
    prior_sigma = 5
    num_data = 50
    num_quad = 30
    poly_order = 3
    multi_q = 10
    x_min = 1
    x_max = 5

    def forward_model(z, x_):
        def sigmoid(y_):
            return np.exp(y_) / (1 + np.exp(y_))
        return 1 - sigmoid((x_ - z)**2 + (z - true_z)**2)

    def err_dens(x_):
        rv = norm(0, noise_sigma)
        return np.prod([rv.pdf(el) for el in x_])

    def prior(x_):
        rv = norm(prior_mean, prior_sigma)
        if type(x_) is np.array:
            return np.prod(np.array([rv.pdf(el) for el in x_]))
        else:
            return rv.pdf(x_)

    x = np.linspace(x_min, x_max, num_data)
    y = forward_model(true_z, x)
    data = y + np.random.normal(0, noise_sigma, len(y))

    hfm = HighFidelityModel(
        forward_model, data, x, err_dens, prior)

    lfm = LowFidelityModel(
        data, err_dens, prior, poly_order, multi_q)

    zs = np.random.normal(prior_mean, prior_sigma, num_quad)
    evals = np.concatenate([hfm.eval(z_).reshape(1, -1) for z_ in zs], axis=0)
    lfm.fit(zs, evals)

    z_min = 2
    z_max = 6
    num_points = 1000

    def plot_surfaces(surrogates, titles=None):
        fig = plt.figure()
        for i, surrogate in enumerate(surrogates):
            a3d = fig.add_subplot(1, len(surrogates), i+1, projection='3d')
            z_finegrid = np.linspace(z_min, z_max, num_points)
            x_finegrid = np.linspace(x_min, x_max, num_points)
            X, Y = np.meshgrid(z_finegrid, x_finegrid)
            Z = np.array(forward_model(X, Y))
            X_, Y_ = np.meshgrid(z_finegrid, x)
            Z_ = np.concatenate([surrogate.eval(z_).reshape(-1, 1) for z_ in z_finegrid], axis=1)
            a3d.plot_surface(X, Y, Z, cmap='viridis')
            a3d.plot_wireframe(X_, Y_, Z_, rstride=5, cstride=100)
            plt.xlabel('z')
            plt.ylabel('x')
            if titles:
                plt.title(titles[i])
        plt.show()

    class UnifProposal:
        def __init__(self, r):
            self.r = r

        def draw(self, z):
            return z + np.random.uniform(-self.r, self.r)

        def density(self, z1, z2):
            return 1./(2 * self.r) if np.abs(z1 - z2) <= self.r else 0

    proposal = UnifProposal(.5)

    samples = 1000
    init_z = 0.
    subchain_len = 100
    upper_thr = 1e-2
    error_thr = 1e-4
    init_radius = 1
    rho_factor = .8

    mfm = deepcopy(lfm)

    hfmh = metropolis_hastings(hfm, proposal, init_z, samples).flatten()
    lfmh = metropolis_hastings(lfm, proposal, init_z, samples).flatten()
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

    plt.figure()

    plt.subplot(1, 3, 1)
    plt.plot(hfmh, label='true model MH')
    plt.plot(lfmh, label='surrogate model MH')
    plt.plot(mfmh, label='adaptive MH')
    plt.legend()

    burn = 200
    plt.subplot(1, 3, 2)
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

    plt.subplot(1, 3, 3)
    plt.plot(x, hfm.eval(true_z), label='true model')
    plt.plot(x, lfm.eval(true_z), label='low-fidelity model')
    plt.plot(x, mfm.eval(true_z), label='multi-fidelity model')
    plt.plot(x, data, label='synthetic data', c='gray')
    plt.legend()
    plt.show()

    plot_surfaces((lfm, mfm), titles=('low-fidelity surrogate', 'multi-fidelity surrogate'))
