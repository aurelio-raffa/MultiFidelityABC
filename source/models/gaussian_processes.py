import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler

from source.models.base_model import ForwardModel


class GPSurrogate(ForwardModel):
    def __init__(self, data, log_error_density, prior, log_prior, kernel, multi_fidelity_q):
        """
        This class creates a surrogate Gaussian Process Model for the forward model G(.)
        :param data: numpy.array, sample of observed values of the forward model
                     in the spatial nodes
        :param log_error_density: function, it computes the log density at the input
                                  vector (used for computing the likelihood, passing
                                  as input the difference between the data and the
                                  vector of the evaluations of the surrogate model
                                  in the corresponding spatial nodes for a fixed
                                  parameter z)
        :param prior: chaospy.Iid, object that represents the prior of the parameter
        :param log_prior: function, it computes the log prior at the input vector
                          (usually vector of parameters)
        :param kernel: kernel instance, hyperparameter of the GPR
        :param multi_fidelity_q: int, it indicates the number of drawn parameter
                                 samples when I update the low fidelity model
        """
        super().__init__()
        self.data = data
        self.log_error_density = log_error_density
        self.prior = prior
        self.log_prior = log_prior
        self.kernel = kernel
        self.multi_fidelity_q = multi_fidelity_q
        self.last_evaluations = [(None, None), (None, None)]
        self.regressors = []
        self.scalers = []
        self._fit = False

    def _fit_subroutine(self, quad_points, true_evals):
        regressor = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True, copy_X_train=False)
        scaler = StandardScaler()
        adj_quad = scaler.fit_transform(quad_points.T)
        regressor.fit(adj_quad, true_evals)
        self.regressors.append(regressor)
        self.scalers.append(scaler)

    def fit(self, high_fidelity, num_evals):
        quad_z = self.prior.sample(num_evals)
        evals = np.concatenate([high_fidelity.eval(z_).reshape(1, -1) for z_ in quad_z.T], axis=0)
        self._fit_subroutine(quad_z, evals)
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
            prediction = np.zeros_like(self.data)
            for regressor, scaler in zip(self.regressors, self.scalers):
                adj_z = scaler.transform(z.reshape(1, -1))
                prediction += regressor.predict(adj_z).reshape((-1,))
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
        self._fit_subroutine(new_points, new_evals)
