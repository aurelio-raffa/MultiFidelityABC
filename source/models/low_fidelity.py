import numpy as np
import matplotlib.pyplot as plt

from .base_model import ForwardModel


class LowFidelityModel(ForwardModel):
    def __init__(self, data, log_error_density, log_prior, degree, multi_fidelity_q):
        super().__init__()
        self.data = data
        self.log_error_density = log_error_density
        self.log_prior = log_prior
        self.degree = degree
        self.multi_fidelity_q = multi_fidelity_q
        self._fit = False
        self.expansion_coeffs = None

    def fit(self, quad_nodes, forward_eval):
        self.expansion_coeffs = np.polynomial.hermite.hermfit(quad_nodes, forward_eval, self.degree)
        self._fit = True

    def eval(self, z):
        assert self._fit
        return np.polynomial.hermite.hermval(z, self.expansion_coeffs)

    def logposterior(self, z):
        predicted = self.eval(z)
        return self.log_error_density(self.data - predicted) * self.log_prior(z)

    def multi_fidelity_update(self, y, radius, high_fidelity):
        if type(y) is np.array:
            new_points = y + np.random.uniform(-radius, radius, (y.shape[0], self.multi_fidelity_q))
        else:
            new_points = y + np.random.uniform(-radius, radius, self.multi_fidelity_q)
        hf_eval = np.concatenate([high_fidelity.eval(z_).reshape(1, -1) for z_ in new_points], axis=0)
        lf_eval = np.concatenate([self.eval(z_).reshape(1, -1) for z_ in new_points], axis=0)
        self.expansion_coeffs += np.polynomial.hermite.hermfit(new_points, hf_eval - lf_eval, self.degree)
