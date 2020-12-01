import chaospy as cpy
import numpy as np

from chaospy import Normal, generate_expansion

from .base_model import ForwardModel


class PCESurrogate(ForwardModel):
    def __init__(self, data, error_density, prior, degree, multi_fidelity_q):
        super().__init__()
        self.data = data
        self.error_density = error_density
        self.prior = prior
        self.degree = degree
        self.multi_fidelity_q = multi_fidelity_q
        self.polynomials = None
        self.expansion = None
        self._fit = False

    def fit(self, high_fidelity, quadrature_rule='gaussian'):
        abscissae, weights = cpy.generate_quadrature(self.degree, self.prior, rule=quadrature_rule)
        self.expansion = generate_expansion(self.degree, self.prior, retall=False)
        evals = np.concatenate([high_fidelity.eval(z_).reshape(-1, 1) for z_ in abscissae.T], axis=1)
        self.polynomials = [
            cpy.fit_quadrature(self.expansion, abscissas, weights, ev) for ev in evals]
        self._fit = True

    def eval(self, z):
        assert self._fit
        return np.array([poly(*z) for poly in self.polynomials])

    def posterior(self, z):
        predicted = self.eval(z)
        return self.error_density(self.data - predicted) * self.prior.pdf(z)

    def multi_fidelity_update(self, y, radius, high_fidelity):
        if type(y) is np.array:
            new_points = y + np.random.uniform(-radius, radius, (y.shape[0], self.multi_fidelity_q))
        else:
            new_points = y + np.random.uniform(-radius, radius, self.multi_fidelity_q)
        hf_eval = np.concatenate([high_fidelity.eval(z_).reshape(-1, 1) for z_ in new_points.T], axis=1)
        lf_eval = np.concatenate([self.eval(z_).reshape(-1, 1) for z_ in new_points.T], axis=1)
        for i, ev in enumerate(hf_eval - lf_eval):
            self.polynomials[i] += cpy.fit_regression(self.expansion, new_points, ev)
