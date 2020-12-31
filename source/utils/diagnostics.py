import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from copy import deepcopy
from mcmc_diagnostics import estimate_ess
from statsmodels.graphics.tsaplots import plot_acf

from source.core.metropolis_hastings import metropolis_hastings, adaptive_multifidelity_mh


def diagnostics_report(
        method_names,
        exec_times,
        fit_times,
        n_lowfi,
        samples,
        burn,
        mh_samples,
        pad_length=35,
        pad_char='.',
        sep_line_width=60):
    print('\nperformance evaluation:')
    messages = ['wall time: ', 'time per iteration: ', 'effective sample size (min.): ', 'fitting time: ']
    formats = [' {:.2f}s', ' {:.4f}s', ' {} / {}', ' {:.4f}s']
    recipes = []
    for msg, form in zip(messages, formats):
        recipes.append(('\n\t\t{' + ':{}<{}'.format(pad_char, pad_length) + '}{}').format(msg, form))

    for i, (name, ex_t, mhs) in enumerate(zip(method_names, exec_times, mh_samples)):
        out_str = '{}\n\t{}:'.format('─' * sep_line_width, name)
        args = [
            [ex_t],
            [ex_t / samples],
            [np.min(np.round(estimate_ess(mhs[:, burn:].T, method='monotone-sequence'))).astype(int),
             samples]]
        if i:
            args.append([fit_times[(i - 1) % n_lowfi]])
        for arg, recipe in zip(args, recipes):
            out_str += recipe.format(*arg)
        print(out_str)


def visual_inspection(dim, method_names, mh_samples, samples, burn, figsize=(15, 10)):
    plt.figure(figsize=figsize)
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
            stat='probability',
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


def run_and_track(
        high_fidelity_model,
        low_fidelity_models,
        quadrature_points,
        proposal,
        var_full_conditional,
        init_z,
        init_sigma,
        samples,
        subchain_len,
        upper_th,
        error_th,
        init_radius,
        rho):
    fit_times = []
    exec_times = []
    mh_samples = []
    mul_fi_models = []
    models = [high_fidelity_model] + low_fidelity_models

    for mod in low_fidelity_models:
        t = mod.fit(high_fidelity_model, quadrature_points)
        fit_times.append(t)
        mul_fi_models.append(deepcopy(mod))

    for mod in models:
        t = time.time()
        mh_samples.append(metropolis_hastings(var_full_conditional, mod, proposal, init_z, init_sigma, samples))
        exec_times.append(time.time() - t)

    for mod in mul_fi_models:
        t = time.time()
        mh_samples.append(
            adaptive_multifidelity_mh(
                subchain_len, samples // subchain_len, upper_th, error_th,
                init_radius, rho,
                mod, high_fidelity_model,
                proposal, init_z, var_full_conditional, init_sigma))
        exec_times.append(time.time() - t)
    return fit_times, exec_times, mh_samples
