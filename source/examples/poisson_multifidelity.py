import fenics

import numpy as np
import chaospy as cpy

from chaospy import Uniform
from sklearn.gaussian_process.kernels import RBF
from scipy.special import logit

from source.models.gaussian_processes import GPSurrogate
from source.models.high_fidelity import HighFidelityModel
from source.models.chaos_expansion import PCESurrogate
from source.problems.poisson_equation_base import PoissonEquation
from source.distributions.cond_inv_gamma import CondInvGamma
from source.distributions.gauss_distr import GaussDensity
from source.distributions.uniform_distr import UnifDistr
from source.utils.diagnostics import run_and_track, diagnostics_report, visual_inspection


def main():
    fenics.set_log_level(30)
    np.random.seed(1226)

    # problem parameters
    dim = 2
    tol = 1e-5                              # tol is used for not drawing nodes from the boundary
    num_data = 100                          # sample's dimension of data collected
    noise_sigma = .3                        # since we generate the data we add some artificial noise
    logit_true_z = np.array([.25, .75])

    # distribution parameters
    prior_means = np.array([0., 0.])
    prior_sigmas = np.array([1.5, 1.5])
    proposal_sigma = .05
    gamma_parameters = np.array([1., 1.])

    # MCMC parameters
    samples = 50000
    subchain_len = 1000
    upper_th = 1e-4
    error_th = 1e-2
    init_z = np.array([.0, .0])
    init_sigma = 1.
    init_radius = .1
    rho = .9
    burn = 2000

    # surrogate parameters
    use_gpr = False
    quad_points = 20
    multi_fidelity_q = 12

    # definition of the forward model
    forward_model = PoissonEquation(
        np.array([32, 32]), 'exp(-100*(pow(x[0] - param0, 2) + pow(x[1] - param1, 2)))',
        np.array([.5, .5]), '0', reparam=True)

    # generation of the dataset
    true_z = logit(logit_true_z)
    x = np.random.uniform(0 + tol, 1 - tol, size=(2, num_data))
    data = forward_model(true_z, x)
    data += np.random.normal(0, noise_sigma, size=data.shape)

    # useful distributions and densities
    def log_prior(z_):
        return -np.sum((z_ - prior_means) ** 2 / (2 * prior_sigmas ** 2))

    prior = cpy.J(*[cpy.Normal(m, s) for m, s in zip(prior_means, prior_sigmas)])
    log_err_density = GaussDensity(1.)
    proposal = GaussDensity(proposal_sigma)
    full_cnd_sigma2 = CondInvGamma(*gamma_parameters)

    # models
    hfm = HighFidelityModel(
        forward_model, data, x, log_err_density, log_prior)
    lfm = PCESurrogate(data, log_err_density, prior, log_prior, 2, multi_fidelity_q)
    gps = GPSurrogate(data, log_err_density, prior, log_prior, RBF(.1), multi_fidelity_q)

    if use_gpr:
        low_fi_models = [lfm, gps]
        surrogate_types = ['PCE', 'GPR']
    else:
        low_fi_models = [lfm]
        surrogate_types = ['PCE']
    method_names = ['true model (MH)'] + \
        ['{} surrogate (MH)'.format(typ) for typ in surrogate_types] + \
        ['{} surr. (adap. MH)'.format(typ) for typ in surrogate_types]

    # running MCMCs
    fit_times, exec_times, mh_samples = run_and_track(
        hfm, low_fi_models, quad_points,
        proposal, full_cnd_sigma2, init_z, init_sigma,
        samples, subchain_len, upper_th, error_th, init_radius, rho)

    # displaying results
    diagnostics_report(method_names, exec_times, fit_times, len(low_fi_models), samples, burn, mh_samples)
    visual_inspection(dim, method_names, mh_samples, samples, burn)


if __name__ == '__main__':
    main()
