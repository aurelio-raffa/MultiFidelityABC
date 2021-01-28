import numpy as np
import progressbar as pb

from source.utils.misc import init_outfile, update_outfile


def _get_log_alpha(model, proposal, z_star, z_prev):
    log_denom = model.logposterior(z_prev) + proposal.logdensity(z_star, z_prev)
    log_numer = model.logposterior(z_star) + proposal.logdensity(z_prev, z_star)
    log_a_ = np.min([0., log_numer - log_denom])
    return log_a_


def metropolis_hastings(
        full_conditional_sigma2,
        model,
        proposal,
        init_z,
        init_variance,
        number_of_samples,
        log=True,
        out_path=None):
    """Metropolis-within-Gibbs MCMC routine based on log-densities.

    Parameters
    ----------
    full_conditional_sigma2 : object
        Object with a `draw` method to sample from the varianceàs full conditional.
    model : ForwardModel
        Instance of a subclass of ForwardModel.
    proposal : object
        Object with a `draw` and a `logdensity` method to (respectively) draw a proposal
        sample and evaluate the log-density at a new point given the previous.
    init_z : float or np.ndarray
        Initial value for the parameter(s) of the forward model.
    init_variance : float
        The initial value of the noise variance.
    number_of_samples : int
        Number of samples to be extracted.
    log : bool, default True
        Whether to display a progress bar for the algorithm containing an estimate of the ETA.
    out_path : str or None, default None
        Path to a .csv file where to store samples of the chain. If None is provided,
        a new timestamped file will be automatically opened

    Returns
    -------
    numpy.ndarray
        A matrix containing the MCMC sample for each parameter and for the variance.
    """
    z_prev = init_z
    variance_prev = init_variance

    if type(init_z) is np.ndarray:
        draws = np.zeros((len(init_z)+1, number_of_samples))
    else:
        draws = np.zeros((2, number_of_samples))
    if log:
        widgets = ['MH\t', pb.Percentage(), ' ', pb.Bar('='), ' ', pb.AdaptiveETA(), ' - ', pb.Timer()]
        bar = pb.ProgressBar(maxval=number_of_samples, widgets=widgets)
        bar.start()
    else:
        bar = None
    if out_path is None:
        out_path = init_outfile(draws.shape[0])
    for iteration in range(number_of_samples):
        z_star = proposal.draw(z_prev)
        model.log_error_density.sigma = variance_prev
        log_alpha = _get_log_alpha(model, proposal, z_star, z_prev)
        z_star = z_star if np.log(np.random.uniform(0, 1)) < log_alpha else z_prev
        draws[:-1, iteration] = z_star
        draws[-1, iteration] = variance_prev
        z_prev = z_star
        variance_prev = full_conditional_sigma2.draw(model, z_prev)
        update_outfile(out_path, z_prev, variance_prev)
        if log:
            bar.update(iteration+1)
    return draws


def _eval_error(model1, model2, y):
    return np.max(np.abs(model1.eval(y) - model2.eval(y)))


def adaptive_multifidelity_mh(
        subchain_length,
        max_iter,
        upper_threshold,
        error_threshold,
        init_radius,
        rho_factor,
        surrogate,
        high_fidelity,
        proposal,
        init_z,
        full_conditional_sigma2,
        init_variance,
        log=True,
        out_path=None):
    """Adaptive Metropolis-within-Gibbs MCMC routine based on log-densities.

    Parameters
    ----------
    subchain_length : int
        The number of MCMC steps to perform on the low-fidelity surrogate before
        a multi-fidelity update.
    max_iter : int
        The number of multi fidelity updates to perform (at most).
    upper_threshold : float
        The threshold for the infinite-norm error under which we reduce the readius of sampling
        for points in the multi-fidelity update.
    error_threshold : float
        Error threshold over which we trigger a multi-fidelity update of the surrogate.
    init_radius : float
        Initial value of the radius within which to sample points in the multi-fidelity update.
    rho_factor : float
        Factor of reduction of the radius every time the error is below the `upper_threshold`.
    surrogate : SurrogateModel
        The surrogate to be used in the subchains and in the multi-fidelity updates.
    high_fidelity : HighFidelityModel
        The true model.
    proposal : object
        Object with a `draw` and a `logdensity` method to (respectively) draw a proposal
        sample and evaluate the log-density at a new point given the previous.
    init_z : float or np.ndarray
        Initial value for the parameter(s) of the forward model.
    full_conditional_sigma2 : object
        Object with a `draw` method to sample from the varianceàs full conditional.
    init_variance : float
        The initial value of the noise variance.
    log : bool, default True
        Whether to display a progress bar for the algorithm containing an estimate of the ETA.
    out_path : str or None, default None
        Path to a .csv file where to store samples of the chain. If None is provided,
        a new timestamped file will be automatically opened

    Returns
    -------
    numpy.ndarray
        A matrix containing the MCMC sample for each parameter and for the variance;
        the dimension of the output matrix is (<num_params> + 1) * (`subchain_length` * `max_iter`).
    """
    z_prev = init_z
    variance_prev = init_variance
    if type(init_z) is np.ndarray:
        draws = np.zeros((len(init_z)+1, max_iter * subchain_length))
    else:
        draws = np.zeros((2, max_iter * subchain_length))
    if log:
        widgets = ['AMH\t', pb.Percentage(), ' ', pb.Bar('='), ' ', pb.AdaptiveETA(), ' - ', pb.Timer()]
        bar = pb.ProgressBar(maxval=max_iter, widgets=widgets)
        bar.start()
    else:
        bar = None
    if out_path is None:
        out_path = init_outfile(draws.shape[0])
    for iteration in range(max_iter):
        subchain = metropolis_hastings(
            full_conditional_sigma2, surrogate, proposal,
            z_prev, variance_prev, subchain_length - 1,
            log=False, out_path=out_path)
        z_prev = subchain[:len(init_z), -1] if type(z_prev) is np.ndarray else subchain[0, -1]
        variance_prev = subchain[-1, -1]
        surrogate.log_error_density.sigma = variance_prev
        high_fidelity.log_error_density.sigma = variance_prev
        z_star = proposal.draw(z_prev)
        log_alpha = _get_log_alpha(high_fidelity, proposal, z_star, z_prev)
        y = z_star if np.log(np.random.uniform(0, 1)) < log_alpha else z_prev
        if _eval_error(high_fidelity, surrogate, y) > error_threshold:
            surrogate.multi_fidelity_update(y, init_radius, high_fidelity)
            if _eval_error(high_fidelity, surrogate, y) < upper_threshold:
                init_radius *= rho_factor
        beta = _get_log_alpha(surrogate, proposal, z_star, z_prev)
        z_star = z_star if np.random.uniform(0, 1) < beta else z_prev
        draws[:, iteration * subchain_length:(iteration+1) * subchain_length - 1] = subchain
        draws[:-1, (iteration+1) * subchain_length - 1] = z_star
        draws[-1, (iteration+1) * subchain_length - 1] = variance_prev
        z_prev = z_star
        variance_prev = full_conditional_sigma2.draw(surrogate, z_prev)
        update_outfile(out_path, z_prev, variance_prev)
        if log:
            bar.update(iteration+1)
    return draws

