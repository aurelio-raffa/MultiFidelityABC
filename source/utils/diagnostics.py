import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from copy import deepcopy
from datetime import datetime
from itertools import product, combinations
from matplotlib import cm
from mcmc_diagnostics import estimate_ess
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.graphics.tsaplots import plot_acf

from source.core.metropolis_hastings import metropolis_hastings, adaptive_multifidelity_mh


def diagnostics_report(
        method_names,
        exec_times,
        exec_calls,
        fit_times,
        fit_calls,
        n_lowfi,
        samples,
        burn,
        mh_samples,
        pad_length=45,
        pad_char='.',
        sep_line_width=60):
    print('\nperformance evaluation:')
    messages = [
        'wall time: ',
        '(avg.) time per iteration: ',
        'effective sample size (min.): ',
        'true model eval.: ',
        'fitting time: ',
        'true mod. eval. during fitting: ']
    formats = [
        ' {:.2f}s', ' {:.4f}s', ' {} / {}', ' {}', ' {:.4f}s', ' {}']
    recipes = []

    for msg, form in zip(messages, formats):
        recipes.append(('\n\t\t{' + ':{}<{}'.format(pad_char, pad_length) + '}{}').format(msg, form))

    for i, (name, ex_t, mhs, ex_c) in enumerate(zip(method_names, exec_times, mh_samples, exec_calls)):
        out_str = '{}\n\t{}:'.format('─' * sep_line_width, name)
        ess = np.min(np.round(estimate_ess(mhs[:, burn:].T, method='monotone-sequence'))).astype(int)
        args = [
            [ex_t],
            [ex_t / samples],
            [ess, samples],
            [ex_c]]
        if i:
            args.append([fit_times[(i - 1) % n_lowfi]])
            args.append([fit_calls[(i - 1) % n_lowfi]])
        for arg, recipe in zip(args, recipes):
            out_str += recipe.format(*arg)
        print(out_str)


def show_or_save(plotname='plot', save=False, show=True):
    if (sys.platform == 'linux' and show) or save:
        now = datetime.now()
        now_to_string = now.strftime("%Y_%m_%d@%H_%M_%S")
        root_dir = '../../images/'
        filename = '{}_{}.jpg'.format(plotname, now_to_string)
        if not os.path.isdir(root_dir):
            os.mkdir(root_dir)
        plt.savefig(os.path.join(root_dir, filename), dpi=600)
        plt.close()
    elif show:
        plt.show()


def visual_inspection(dim, method_names, mh_samples, samples, burn, figsize=(15, 10), save=False):
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
            kde=True,
            line_kws={'linestyle': 'dashed'},
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

    show_or_save(save=save)


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
        rho,
        remap_functions=None):
    fit_times = []
    fit_calls = []
    exec_times = []
    exec_calls = []
    mh_samples = []
    mul_fi_models = []
    models = [high_fidelity_model] + low_fidelity_models
    solver = high_fidelity_model.core_function.solver

    for mod in low_fidelity_models:
        solver.reset_calls()
        t = mod.fit(high_fidelity_model, quadrature_points)
        fit_times.append(t)
        fit_calls.append(solver.get_calls())
        mul_fi_models.append(deepcopy(mod))

    for mod in models:
        solver.reset_calls()
        t = time.time()
        mh_samples.append(
            metropolis_hastings(
                var_full_conditional, mod, proposal, init_z, init_sigma, samples))
        exec_times.append(time.time() - t)
        exec_calls.append(solver.get_calls())

    for mod in mul_fi_models:
        solver.reset_calls()
        t = time.time()
        mh_samples.append(
            adaptive_multifidelity_mh(
                subchain_len, samples // subchain_len, upper_th, error_th,
                init_radius, rho, mod, high_fidelity_model,
                proposal, init_z, var_full_conditional, init_sigma))
        exec_times.append(time.time() - t)
        exec_calls.append(solver.get_calls())

    if remap_functions is not None:
        for mhd in mh_samples:
            for i, fun in enumerate(remap_functions):
                if fun is not None:
                    mhd[i, :] = fun(mhd[i, :])

    return fit_times, fit_calls, exec_times, exec_calls, mh_samples


def _get_3dwindow(figsize=(10, 10), angles=None):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection='3d')
    if angles:
        ax.view_init(*angles)
    ax.grid(False)
    return fig, ax


def _base_3d_plotter(x_ranges, y_ranges, fun, step=.1, color_fun=None, angles=None):
    xx = np.arange(*x_ranges, step)
    yy = np.arange(*y_ranges, step)
    xx, yy = np.meshgrid(xx, yy)
    zz = fun(xx, yy)

    if color_fun is not None:
        colors = cm.magma(color_fun(xx, yy))
        cmap = 'magma'
    else:
        colors = None
        cmap = 'viridis'

    fig, ax = _get_3dwindow(angles=angles)
    return fig, ax, xx, yy, zz, cmap, colors


def surface_plot(x_ranges, y_ranges, fun, step=.1, color_fun=None, angles=None, show=True, colorbar=True, save=False):
    fig, ax, xx, yy, zz, cmap, colors = _base_3d_plotter(
        x_ranges, y_ranges, fun, step=step, color_fun=color_fun, angles=angles)
    surf = ax.plot_surface(xx, yy, zz, cmap=cmap, linewidth=0, antialiased=False, facecolors=colors)
    if colorbar:
        fig.colorbar(surf, shrink=0.3)

    show_or_save(plotname='surface', save=save, show=show)
    return fig, ax


def wireframe_plot(
        x_ranges, y_ranges, fun, stride=5, step=.1, color_fun=None, angles=None, show=True, colorbar=False, save=False):
    fig, ax, xx, yy, zz, cmap, colors = _base_3d_plotter(
        x_ranges, y_ranges, fun, step=step, color_fun=color_fun, angles=angles)
    wire = ax.plot_wireframe(xx, yy, zz, alpha=.5, rstride=stride, cstride=stride, color='gray')
    if colorbar:
        fig.colorbar(wire, shrink=0.3)

    show_or_save(plotname='wireframe', save=save, show=show)
    return fig, ax


def points_plot(
        fig, ax, coords, values, angles=None, color=None, colorbar=True, show=True, save=False, **scatter_kwargs):
    if angles:
        ax.view_init(*angles)
    ax.grid(False)
    points = ax.scatter(coords[0, :], coords[1, :], values, c=color, cmap='coolwarm', **scatter_kwargs)

    if colorbar:
        fig.colorbar(points, shrink=0.3)

    show_or_save(plotname='points', save=save, show=show)
    return fig, ax, points


def box3d_plot(vtx1, vtx2, **plot3d_kwargs):
    r = [np.sort(pair) for pair in zip(vtx1, vtx2)]
    fig, ax = _get_3dwindow()
    for s, e in combinations(np.array(list(product(*r))), 2):
        if np.sum(np.abs(s - e)) in [r_[1] - r_[0] for r_ in r]:
            ax.plot3D(*zip(s, e), color='black', **plot3d_kwargs)
    return fig, ax
