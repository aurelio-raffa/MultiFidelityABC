import numpy as np
import progressbar as pb

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler

from source.models.surrogate import SurrogateModel
from source.utils.decorators import time_it


class GPSurrogate(SurrogateModel):
    """Implements a low-fidelity surrogate approximating the forward
    model through Gaussian Process Regression.
    """
    def __init__(self, data, log_error_density, prior, log_prior, kernel, multi_fidelity_q, **gpr_kwargs):
        """Parameters
        ----------
        data : numpy.ndarray
            Sample of observed values of the forward model in the spatial nodes.
        log_error_density : function or callable object
            It computes the log density at the input vector (used for computing the likelihood, passing
            as input the difference between the data and the vector of the evaluations of the surrogate model
            in the corresponding spatial nodes for a fixed  parameter z).
        prior : object
            Object that represents the prior of the parameter from chaospy.distributions.
        log_prior : function or callable object
            It computes the log prior at the input vector (of parameters).
        kernel : object
            The kernel to be fed to scikit.gaussian_process.GaussianProcessRegressor
        multi_fidelity_q : int
            Number of points to be drawn when performing a multi-fidelity update.
        **gpr_kwargs : optional
            Keyword additional arguments to be passed to scikit.gaussian_process.GaussianProcessRegressor
        """
        super().__init__(data, log_error_density, prior, log_prior, multi_fidelity_q)
        self.kernel = kernel
        self.regressors = []
        self.scalers = []
        self._gpr_kwargs = gpr_kwargs

    def _fit_subroutine(self, quad_points, true_evals):
        regressor = GaussianProcessRegressor(
            kernel=self.kernel, normalize_y=True, copy_X_train=True, **self._gpr_kwargs)
        scaler = StandardScaler()
        adj_quad = scaler.fit_transform(quad_points.T)
        regressor.fit(adj_quad, true_evals)
        self.regressors.append(regressor)
        self.scalers.append(scaler)

    @time_it(only_time=True)
    def fit(self, high_fidelity, num_evals):
        """Fits the low-fidelity surrogate via Gaussian Process Regression.

        Parameters
        ----------
        high_fidelity : HighFidelityModel
            Model that we want to approximate through the low-fidelity.
        num_evals : int
            Number of evaluations required to construct the low-fidelity surrogate
            (sampled randomly from the prior).
        """
        quad_z = self.prior.sample(num_evals)
        if quad_z.ndim == 1:
            quad_z = quad_z.reshape(1, -1)
        widgets = ['fit\t', pb.Percentage(), ' ', pb.Bar('='), ' ', pb.AdaptiveETA(), ' - ', pb.Timer()]
        bar = pb.ProgressBar(maxval=quad_z.T.shape[0], widgets=widgets)
        evals = []
        bar.start()
        for i, z_ in enumerate(quad_z.T):
            evals.append(high_fidelity.eval(z_).reshape(1, -1))
            bar.update(i + 1)
        evals = np.concatenate(evals, axis=0)
        self._fit_subroutine(quad_z, evals)
        self._fit = True

    def _eval_subroutine(self, z):
        prediction = np.zeros_like(self.data)
        for regressor, scaler in zip(self.regressors, self.scalers):
            adj_z = scaler.transform(z.reshape(1, -1) if type(z) is np.ndarray else np.array(z).reshape(1, -1))
            prediction += regressor.predict(adj_z).reshape((-1,))
        return prediction

    def multi_fidelity_update(self, y, radius, high_fidelity):
        new_points = super().multi_fidelity_update(y, radius, high_fidelity)
        new_evals = np.concatenate([
            (high_fidelity.eval(z_) - self.eval(z_)).reshape(1, -1) for z_ in new_points.T], axis=0)
        self._fit_subroutine(new_points, new_evals)
