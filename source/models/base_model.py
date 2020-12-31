import numpy as np


class ForwardModel:
    def __init__(self, data, log_error_density, log_prior):
        self.data = data
        self.log_error_density = log_error_density
        self.log_prior = log_prior
        self.last_evaluations = [(None, None), (None, None)]

    def _eval_subroutine(self, z):
        pass

    def eval(self, z):
        if self.last_evaluations[0][0] is not None and np.all(
                np.abs(z - self.last_evaluations[0][0]) < np.finfo(np.float32).eps):
            pass
        # condition to avoid repeated evaluations for the same parameter
        elif self.last_evaluations[1][0] is not None and np.all(
                np.abs(z - self.last_evaluations[1][0]) < np.finfo(np.float32).eps):
            self.last_evaluations.reverse()
        else:
            self.last_evaluations[1] = (z, self._eval_subroutine(z))
            self.last_evaluations.reverse()
        return self.last_evaluations[0][1]

    def logposterior(self, z):
        predicted = self.eval(z)
        res = self.log_error_density(self.data - predicted) + self.log_prior(z)
        return res


