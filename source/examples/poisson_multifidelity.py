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
from sklearn.gaussian_process.kernels import RBF

from source.models.gaussian_processes import GPSurrogate
from source.models.high_fidelity import HighFidelityModel
from source.models.chaos_expansion import PCESurrogate
from source.core.metropolis_hastings import metropolis_hastings, adaptive_multifidelity_mh
from source.problems.poisson_equation_base import PoissonEquation
from source.distributions.cond_inv_gamma import CondInvGamma
from source.distributions.gauss_distr import GaussDensity
from source.distributions.uniform_distr import UnifDistr
from statsmodels.graphics.tsaplots import plot_acf


if __name__ == '__main__':
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
    noise_sigma = .25  # since we generate the data we add...
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

    proposal = UnifDistr(.01, tol)
    samples = 2000
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

    # per l'hfm dovrò fare metropolis-hasting usando la vera posterior, proporzionale a
    # prior*likelihood, dove la likelihood è il prodotto delle err_dens nei punti d_i-G(x_i;z)
    # perciò ad ogni passo del mh dovrò calcolare per il parametro z proposto la soluzione
    # della pde nei vari nodi x_i (tipicamente operazione costosa la risoluzione della pde).
    # passerò come parametri la funzione (intesa come la soluzione della pde), i dati
    # che serviranno per il calcolo della likelihood, i nodi su cui andrò a valutare la funzione
    # error_dens che mi servirà per calcolo della likelihood, e prior che servirà per calcolare
    # la posterior (queste ultime come log sempre per motivi di instabilità numerica)

    # definisco il low fidelity model che sarà un'approssimazione poolinomiale, della funzione vera
    # (soluzione della pde), che stimo a partire dai dati.
    # non mi è chiaro perché passo sia prior sia log_prior
    # poi secifico il grado dei polinomi approssimanti ed il multi_fidelity_q
    # PCESurrogate fa la stessa cosa cosa che fa la classe LowFidelityModel?

    hfm = HighFidelityModel(
        forward_model, data, x, log_err_density, log_prior)
    quad_points = 10
    multi_fidelity_q = 12
    lfm = PCESurrogate(data, log_err_density, prior, log_prior, 2, multi_fidelity_q)
    gps = GPSurrogate(data, log_err_density, prior, log_prior, RBF(.1), multi_fidelity_q)

    low_fi_models = [lfm, gps]
    surrogate_types = ['PCE', 'GPR']
    models = [hfm] + low_fi_models
    mul_fi_models = []
    method_names = ['true model (MH)'] + \
        ['{} surrogate (MH)'.format(typ) for typ in surrogate_types] + \
        ['{} surr. (adap. MH)'.format(typ) for typ in surrogate_types]

    fit_times = []
    for mod in low_fi_models:
        t = time.time()
        mod.fit(hfm, quad_points)
        fit_times.append(time.time() - t)
        mul_fi_models.append(deepcopy(mod))

    exec_times = []
    mh_samples = []
    for mod in models:
        t = time.time()
        mh_samples.append(metropolis_hastings(full_cnd_sigma2, mod, proposal, init_z, init_sigma, samples))
        exec_times.append(time.time() - t)

    for mod in mul_fi_models:
        t = time.time()
        mh_samples.append(adaptive_multifidelity_mh(
            subchain_len,
            samples // subchain_len,
            upper_th,
            error_th,
            init_radius,
            rho,
            mod,
            hfm,
            proposal,
            init_z,
            full_cnd_sigma2,
            init_sigma))
        exec_times.append(time.time() - t)

    print('\nperformance evaluation:')
    for i, (name, ex_t) in enumerate(zip(method_names, exec_times)):
        out_str = '\t{}:\t{:.2f}s ({:.4f}s per iteration) '.format(name, ex_t, ex_t/samples)
        out_str += '' if not i else '[{:.4f}s fitting time]'.format(fit_times[(i - 1) % len(low_fi_models)])
        print(out_str)

    plt.figure(figsize=(15, 10))
    plot_names = ['parameter {}'.format(i + 1) for i in range(dim)] + ['variance']
    cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i, pname in enumerate(plot_names):
        plt.subplot(3, dim + 1, i + 1)
        for j, (dname, mhdata) in enumerate(zip(method_names, mh_samples)):
            plt.plot(mhdata[i, :], label=dname, alpha=.3)
            cmean = np.cumsum(mhdata[i, :]) / np.arange(1, mhdata.shape[1] + 1)
            plt.plot(cmean, ':', color=cycle[j])
        plt.title(pname)
        plt.legend()

    burn = 400
    for i, pname in enumerate(plot_names):
        plt.subplot(3, dim + 1, i + dim + 2)
        plot_data = np.concatenate([
            mhdata[i, burn:] for mhdata in mh_samples],
            axis=0)
        labels = []
        for dname in method_names:
            labels += [dname] * (samples - burn)
        mh_data = pd.DataFrame({
            'data': plot_data,
            'method': labels})
        sns.histplot(
            mh_data,
            x='data',
            hue='method',
            stat='density',
            bins=75,
            multiple='layer',
            edgecolor='.3',
            linewidth=.5)
        plt.xlabel('')
        plt.ylabel('')
        plt.title(pname)

    for i, pname in enumerate(plot_names):
        ax = plt.subplot(3, dim + 1, i + 2 * dim + 3)
        for dname, mhdata in zip(method_names, mh_samples):
            plot_acf(mhdata[i, burn:], ax=ax, alpha=None, label=dname, marker='.', vlines_kwargs={'color': 'lightgray'})
        handles, labels = ax.get_legend_handles_labels()
        handles = handles[1::2]
        labels = labels[1::2]
        ax.legend(handles=handles, labels=labels)
        plt.title(pname)

    if sys.platform == 'linux':
        plt.savefig('../../images/plots.jpg')
        plt.close()
    else:
        plt.show()

# guardando i plot perché gli istogrammi del true model e dell'adaptive non sono centrati
# nei true value dei parametri?
