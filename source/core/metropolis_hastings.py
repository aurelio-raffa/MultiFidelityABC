import numpy as np


def __get_alpha(model, proposal, z_star, z_prev):
    a_ = 1. / (model.posterior(z_prev) * proposal.density(z_star, z_prev))
    a_ *= model.posterior(z_star) * proposal.density(z_prev, z_star)
    return np.min([1., a_])


def metropolis_hastings(
        surrogate,
        proposal,
        init_z,
        number_of_samples):

    z_prev = init_z
    if type(init_z) is np.array:
        draws = np.zeros((len(init_z), number_of_samples))
    else:
        draws = np.zeros((1, number_of_samples))
    for iteration in range(number_of_samples):
        z_star = proposal.draw(z_prev)
        alpha = __get_alpha(surrogate, proposal, z_star, z_prev)
        z_star = z_star if np.random.uniform(0, 1) < alpha else z_prev
        draws[:, iteration] = z_star
        z_prev = z_star
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
        init_z):

    z_prev = z_init
    draws = np.zeros((len(init_z), max_iter * subchain_length))
    for iteration in range(max_iter):
        subchain = metropolis_hastings(surrogate, proposal, z_prev, subchain_length - 1)
        z_prev = subchain[-1]
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
    return draws

