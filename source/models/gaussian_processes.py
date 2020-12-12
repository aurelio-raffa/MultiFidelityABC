import numpy as np

from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from .base_model import ForwardModel


class GPSurrogate(ForwardModel):

    def __init__(self, data, log_error_density, prior, log_prior, theta, lenscale, multi_fidelity_q):
        super().__init__()
        self.data = data
        self.log_error_density = log_error_density
        self.prior = prior
        self.log_prior = log_prior
        self.theta = theta
        self.lenscale = lenscale
        self.multi_fidelity_q = multi_fidelity_q
        self.kernel = ConstantKernel(theta) * RBF(lenscale)
        self.last_evaluations = [(None, None), (None, None)]
        self.regressor = GaussianProcessRegressor(kernel=self.kernel)
        self.multifidelity_regressors = []
        self._fit = False

    def fit(self, high_fidelity, num_evals):
        quad_z = self.prior.sample(num_evals).T
        evals = np.concatenate([high_fidelity.eval(z_).reshape(1, -1) for z_ in quad_z], axis=0)
        self.regressor.fit(quad_z, evals)
        self._fit = True

    def eval(self, z):
        assert self._fit
        if self.last_evaluations[0][0] is not None and np.all(
                np.abs(z - self.last_evaluations[0][0]) < np.finfo(np.float32).eps):
            pass
        # condition to avoid repeated evaluations for the same parameter
        elif self.last_evaluations[1][0] is not None and np.all(
                np.abs(z - self.last_evaluations[1][0]) < np.finfo(np.float32).eps):
            self.last_evaluations.reverse()
        else:
            prediction = self.regressor.predict(z.reshape(1, -1))
            for regressor in self.multifidelity_regressors:
                prediction += regressor.predict(z.reshape(1, -1))
            self.last_evaluations[1] = (z, prediction)
            self.last_evaluations.reverse()
        return self.last_evaluations[0][1]

    def logposterior(self, z):
        predicted = self.eval(z)
        res = self.log_error_density(self.data - predicted) + self.log_prior(z)
        return res

    def multi_fidelity_update(self, y, radius, high_fidelity):
        if type(y) is np.ndarray:
            new_points = y.reshape(-1, 1) + np.random.uniform(-radius, radius, (y.shape[0], self.multi_fidelity_q))
        else:
            new_points = y + np.random.uniform(-radius, radius, self.multi_fidelity_q)
        new_evals = np.concatenate([
            (high_fidelity.eval(z_) - self.eval(z_)).reshape(1, -1) for z_ in new_points.T], axis=0)
        new_regressor = GaussianProcessRegressor(kernel=ConstantKernel(self.theta)*RBF(self.lenscale))
        new_regressor.fit(new_points.T, new_evals)
        self.multifidelity_regressors.append(new_regressor)
