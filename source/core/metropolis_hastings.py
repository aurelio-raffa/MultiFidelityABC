import numpy as np
import progressbar as pb


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
        log=True):

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
    for iteration in range(number_of_samples):
        z_star = proposal.draw(z_prev)
        model.log_error_density.sigma = variance_prev
        log_alpha = _get_log_alpha(model, proposal, z_star, z_prev)
        z_star = z_star if np.log(np.random.uniform(0, 1)) < log_alpha else z_prev
        draws[:-1, iteration] = z_star
        draws[-1, iteration] = variance_prev
        z_prev = z_star
        variance_prev = full_conditional_sigma2.draw(model, z_prev)
        if log:
            bar.update(iteration+1)
    return draws


def __eval_error(model1, model2, y):
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
        log=True):

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
    for iteration in range(max_iter):
        subchain = metropolis_hastings(
            full_conditional_sigma2, surrogate, proposal, z_prev, variance_prev, subchain_length - 1, log=False)
        z_prev = subchain[:len(init_z), -1] if type(z_prev) is np.ndarray else subchain[0, -1]
        variance_prev = subchain[-1, -1]
        surrogate.log_error_density.sigma = variance_prev
        high_fidelity.log_error_density.sigma = variance_prev
        z_star = proposal.draw(z_prev)
        log_alpha = _get_log_alpha(high_fidelity, proposal, z_star, z_prev)
        y = z_star if np.log(np.random.uniform(0, 1)) < log_alpha else z_prev
        if __eval_error(high_fidelity, surrogate, y) > error_threshold:
            surrogate.multi_fidelity_update(y, init_radius, high_fidelity)
            if __eval_error(high_fidelity, surrogate, y) < upper_threshold:
                init_radius /= rho_factor
        beta = _get_log_alpha(surrogate, proposal, z_star, z_prev)
        z_star = z_star if np.random.uniform(0, 1) < beta else z_prev
        draws[:, iteration * subchain_length:(iteration+1) * subchain_length - 1] = subchain
        draws[:-1, (iteration+1) * subchain_length - 1] = z_star
        draws[-1, (iteration+1) * subchain_length - 1] = variance_prev
        z_prev = z_star
        variance_prev = full_conditional_sigma2.draw(surrogate, z_prev)
        if log:
            bar.update(iteration+1)
    return draws

