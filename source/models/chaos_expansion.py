import chaospy as cpy
import numpy as np

from chaospy import Normal, generate_expansion

from .base_model import ForwardModel


class PCESurrogate(ForwardModel):
    def __init__(self, data, log_error_density, prior, log_prior, degree, multi_fidelity_q):
        super().__init__()
        self.data = data
        self.log_error_density = log_error_density
        self.prior = prior
        self.log_prior = log_prior
        self.degree = degree
        self.multi_fidelity_q = multi_fidelity_q
        self.expansion = None
        self.last_evalutations = [(None, None), (None, None)]
        self.proxy = None
        self._fit = False

    @staticmethod
    def _get_coeffs(fitted_poly):
        return np.concatenate([np.array(p.coefficients).reshape(1, -1) for p in fitted_poly], axis=0)

    def fit(self, high_fidelity, quadrature_rule='gaussian'):
        abscissae, weights = cpy.generate_quadrature(self.degree, self.prior, rule=quadrature_rule)
        self.expansion = generate_expansion(self.degree, self.prior, retall=False)
        evals = [high_fidelity.eval(z_) for z_ in abscissae.T]
        self.proxy = cpy.fit_quadrature(self.expansion, abscissae, weights, evals)
        self._fit = True

    def eval(self, z):
        assert self._fit
        if self.last_evalutations[0][0] is not None and np.all(
                np.abs(z - self.last_evalutations[0][0]) < np.finfo(np.float32).eps):
            pass
        elif self.last_evalutations[1][0] is not None and np.all(
                np.abs(z - self.last_evalutations[1][0]) < np.finfo(np.float32).eps):
            self.last_evalutations.reverse()
        else:
            self.last_evalutations[1] = (z, self.proxy(*z))
            self.last_evalutations.reverse()
        return self.last_evalutations[0][1]

    def logposterior(self, z):
        predicted = self.eval(z)
        res = self.log_error_density(self.data - predicted) + self.log_prior(z)
        return res

    def multi_fidelity_update(self, y, radius, high_fidelity):
        if type(y) is np.ndarray:
            new_points = y.reshape(-1, 1) + np.random.uniform(-radius, radius, (y.shape[0], self.multi_fidelity_q))
        else:
            new_points = y + np.random.uniform(-radius, radius, self.multi_fidelity_q)
        new_evals = [high_fidelity.eval(z_) - self.eval(z_) for z_ in new_points.T]
        new_poly = cpy.fit_regression(self.expansion, new_points, new_evals)
        self.proxy = self.proxy + new_poly
