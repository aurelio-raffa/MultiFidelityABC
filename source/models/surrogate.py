import numpy as np

from source.models.base_model import ForwardModel


class SurrogateModel(ForwardModel):
    def __init__(self, data, log_error_density, prior, log_prior, multi_fidelity_q):
        super().__init__(data, log_error_density, log_prior)
        self.prior = prior
        self.multi_fidelity_q = multi_fidelity_q
        self._fit = False

    def eval(self, z):
        assert self._fit
        return super().eval(z)

    def multi_fidelity_update(self, y, radius, high_fidelity):
        if type(y) is np.ndarray:
            new_points = y.reshape(-1, 1) + np.random.uniform(-radius, radius, (y.shape[0], self.multi_fidelity_q))
        else:
            new_points = y + np.random.uniform(-radius, radius, self.multi_fidelity_q)
        return new_points
