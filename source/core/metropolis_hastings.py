import numpy as np
import progressbar as pb


def __get_alpha(model, proposal, z_star, z_prev):
    log_denom = model.logposterior(z_prev) + proposal.logdensity(z_star, z_prev)
    log_numer = model.logposterior(z_star) + proposal.logdensity(z_prev, z_star)
    a_ = np.min([1., np.exp(log_numer - log_denom)])
    return a_


def metropolis_hastings(
        model,
        proposal,
        init_z,
        number_of_samples,
        log=True):

    z_prev = init_z
    if type(init_z) is np.ndarray:
        draws = np.zeros((len(init_z), number_of_samples))
    else:
        draws = np.zeros((1, number_of_samples))
    if log: #cosa faccio dentro questo if?
        widgets = [pb.Percentage(), ' ', pb.Bar('='), ' ', pb.AdaptiveETA(), ' - ', pb.Timer()]
        bar = pb.ProgressBar(maxval=number_of_samples, widgets=widgets)
        bar.start()
    for iteration in range(number_of_samples):
        z_star = proposal.draw(z_prev)
        alpha = __get_alpha(model, proposal, z_star, z_prev)
        z_star = z_star if np.random.uniform(0, 1) < alpha else z_prev
        draws[:, iteration] = z_star
        z_prev = z_star
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
        log=True):

    z_prev = init_z
    if type(init_z) is np.ndarray:
        draws = np.zeros((len(init_z), max_iter * subchain_length))
    else:
        draws = np.zeros((1, max_iter * subchain_length))
    if log:
        widgets = [pb.Percentage(), ' ', pb.Bar('='), ' ', pb.AdaptiveETA(), ' - ', pb.Timer()]
        bar = pb.ProgressBar(maxval=max_iter, widgets=widgets)
        bar.start()
    for iteration in range(max_iter):
        subchain = metropolis_hastings(surrogate, proposal, z_prev, subchain_length - 1, log=False)
        z_prev = subchain[:, -1] if type(z_prev) is np.ndarray else subchain[0, -1]
        z_star = proposal.draw(z_prev)
        alpha = __get_alpha(high_fidelity, proposal, z_star, z_prev)
        y = z_star if np.random.uniform(0, 1) < alpha else z_prev
        if __eval_error(high_fidelity, surrogate, y) > error_threshold:
            surrogate.multi_fidelity_update(y, init_radius, high_fidelity)
            if __eval_error(high_fidelity, surrogate, y) < upper_threshold:
                init_radius /= rho_factor
        beta = __get_alpha(surrogate, proposal, z_star, z_prev)
        z_star = z_star if np.random.uniform(0, 1) < beta else z_prev
        draws[:, iteration * subchain_length:(iteration+1) * subchain_length - 1] = subchain
        draws[:, (iteration+1) * subchain_length - 1] = z_star
        z_prev = z_star
        if log:
            bar.update(iteration+1)
    return draws

