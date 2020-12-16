import fenics
import time
import sys

import seaborn as sns
import pandas as pd
import numpy as np
import chaospy as cpy
import matplotlib.pyplot as plt

from copy import deepcopy
from chaospy import Uniform

from source.models.high_fidelity import HighFidelityModel
from source.models.chaos_expansion import PCESurrogate
from source.core.metropolis_hastings import metropolis_hastings, adaptive_multifidelity_mh
from source.problems.poisson_equation_base import PoissonEquation
from source.distributions.cond_inv_gamma import CondInvGamma
from source.distributions.gauss_distr import GaussDensity
from source.distributions.uniform_distr import UnifDistr


if __name__ == '__main__':
    # che cosa fa set_log_level?
    fenics.set_log_level(30)
    np.random.seed(1226)

    # la G del paper sarà la soluz dell'eq di Poisson nel quadrato unitario usando griglia 32x32
    # funzione f diperndente dai parametri [.5,.5], definita come l'espressione in verde,
    # 0 credo stia per dire condizioni omogenee di dirichlet
    forward_model = PoissonEquation(
        np.array([32, 32]), 'exp(-100*(pow(x[0] - param0, 2) + pow(x[1] - param1, 2)))',
        np.array([.5, .5]), '0')

    tol = 1e-5  # tol is used for not drawing nodes from the boundary
    num_data = 100  # sample's dimension of data collected
    noise_sigma = .75  # since we generate the data we add...
    true_z = np.array([.25, .75])
    # perché faccio l'uniforme tra [0+tol, 1-tol]
    x = np.random.uniform(0 + tol, 1 - tol, size=(2, num_data))
    true_vals = forward_model(true_z, x)
    data = true_vals + np.random.normal(0, noise_sigma, size=true_vals.shape)

    # dimensione del parametro (in questo caso bidimensionale)
    dim = 2

    # dovrebbe essere f(z1,z2)=f(z1)f(z2) dove f(s) = pdf U(tol,1-tol)
    prior = cpy.Iid(Uniform(0 + tol, 1 - tol), dim)

    # definisco log prior altrimenti avrei problemi di instabilità numerica
    def log_prior(z_):
        # min e max servono solo perché se passo un valore maggiore di 1-tol
        # o minore di tol
        z_ = np.min([z_, np.ones_like(z_) - tol], axis=0)
        z_ = np.max([z_, np.zeros_like(z_) + tol], axis=0)
        return np.log(prior.pdf(z_))

    log_err_density = GaussDensity(1)

    # per l'hfm dovrò fare metropolis-hasting usando la vera posterior, proporzionale a
    # prior*likelihood, dove la likelihood è il prodotto delle err_dens nei punti d_i-G(x_i;z)
    # perciò ad ogni passo del mh dovrò calcolare per il parametro z proposto la soluzione
    # della pde nei vari nodi x_i (tipicamente operazione costosa la risoluzione della pde).
    # passerò come parametri la funzione (intesa come la soluzione della pde), i dati
    # che serviranno per il calcolo della likelihood, i nodi su cui andrò a valutare la funzione
    # error_dens che mi servirà per calcolo della likelihood, e prior che servirà per calcolare
    # la posterior (queste ultime come log sempre per motivi di instabilità numerica)
    hfm = HighFidelityModel(
        forward_model, data, x, log_err_density, log_prior)

    # definisco il low fidelity model che sarà un'approssimazione poolinomiale, della funzione vera
    # (soluzione della pde), che stimo a partire dai dati.
    # non mi è chiaro perché passo sia prior sia log_prior
    # poi secifico il grado dei polinomi approssimanti ed il multi_fidelity_q
    # PCESurrogate fa la stessa cosa cosa che fa la classe LowFidelityModel?
    lfm = PCESurrogate(data, log_err_density, prior, log_prior, 2, 10)
    lfm.fit(hfm)

    proposal = UnifDistr(.05, tol)
    samples = 10000
    init_z = np.array([.5, .5])
    full_cnd_sigma2 = CondInvGamma(1, 1)
    init_sigma = 1

    # questo è l'm del paper, ossia il numero di iterazioni che faccio all'interno del mh dentro
    # il multifidelity
    subchain_len = 100
    upper_th = 1e-4
    error_th = 1e-2
    init_radius = .1
    rho = .9
    mfm = deepcopy(lfm)

    lfmh_t = time.time()
    lfmh = metropolis_hastings(full_cnd_sigma2, lfm, proposal, init_z, init_sigma, samples)
    lfmh_t = time.time() - lfmh_t

    hfmh_t = time.time()
    hfmh = metropolis_hastings(full_cnd_sigma2, hfm, proposal, init_z, init_sigma, samples)
    hfmh_t = time.time() - hfmh_t

    mfmh_t = time.time()
    # quante sono nel multifidelity il numero di z estratte alle fine? sempre samples?
    # sembrerebbe di si
    mfmh = adaptive_multifidelity_mh(
        subchain_len,
        samples // subchain_len,
        upper_th,
        error_th,
        init_radius,
        rho,
        mfm,
        hfm,
        proposal,
        init_z,
        full_cnd_sigma2,
        init_sigma)
    mfmh_t = time.time() - mfmh_t

    print('\nperformance evaluation:')
    print('\tlow fidelity:\t{:.2f}s ({:.4}s per iteration)'.format(lfmh_t, lfmh_t / samples))
    print('\thigh fidelity:\t{:.2f}s ({:.4}s per iteration)'.format(hfmh_t, hfmh_t / samples))
    print('\tmulti fidelity:\t{:.2f}s ({:.4}s per iteration)'.format(mfmh_t, mfmh_t / samples))

    plt.figure(figsize=(15, 10))

    for i in range(3):
        plt.subplot(2, 3, i + 1)

        plt.plot(hfmh[i, :], label='true model MH')
        plt.plot(lfmh[i, :], label='low-fidelity model MH')
        plt.plot(mfmh[i, :], label='multi-fidelity model MH')
        plt.legend()

    burn = 400
    for i in range(3):
        plt.subplot(2, 3, i + 4)

        mh_data = pd.DataFrame({
            'data': np.concatenate([hfmh[i, burn:], lfmh[i, burn:], mfmh[i, burn:]], axis=0),
            'method':
                ['true model MH samples'] * (samples - burn) +
                ['low-fidelity model MH samples'] * (samples - burn) +
                ['multi-fidelity model MH samples'] * (samples - burn)})
        sns.histplot(
            mh_data,
            x='data',
            hue='method',
            bins=50,
            multiple='layer',
            edgecolor='.3',
            linewidth=.5)

    if sys.platform == 'linux':
        plt.savefig('../../images/plots.jpg')
        plt.close()
    else:
        plt.show()

# guardando i plot perché gli istogrammi del true model e dell'adaptive non sono centrati
# nei true value dei parametri?
