import numpy as np

from .base_model import ForwardModel


class HighFidelityModel(ForwardModel):
    def __init__(self, core_function, data, evaluation_nodes, log_error_density, log_prior):
        # make sure that the data nodes are a subset of the quadrature nodes
        super().__init__()
        self.core_function = core_function
        self.data = data
        self.evaluation_nodes = evaluation_nodes
        self.log_error_density = log_error_density
        self.log_prior = log_prior
        self.last_evalutations = [(None, None), (None, None)]       # always keep the last two evaluations in memory

    def eval(self, z):
        if self.last_evalutations[0][0] is not None and np.all(
                np.abs(z - self.last_evalutations[0][0]) < np.finfo(np.float32).eps):
            pass
        elif self.last_evalutations[1][0] is not None and np.all(
                np.abs(z - self.last_evalutations[1][0]) < np.finfo(np.float32).eps):
            self.last_evalutations.reverse()
        else:
            self.last_evalutations[1] = (z, self.core_function(z, self.evaluation_nodes))
            self.last_evalutations.reverse()
        return self.last_evalutations[0][1]     # self.last_evalutations[0] is the most recent entry

    def logposterior(self, z):
        predicted = self.eval(z)
        res = self.log_error_density(self.data - predicted) + self.log_prior(z)       # implement it via mpi
        return res




