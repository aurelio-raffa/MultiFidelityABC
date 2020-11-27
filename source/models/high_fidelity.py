import numpy as np

from .base_model import ForwardModel


class HighFidelityModel(ForwardModel):
    def __init__(self, core_function, data, evaluation_nodes, error_density, prior):
        # make sure that the data nodes are a subset of the quadrature nodes
        super().__init__()
        self.core_function = core_function
        self.data = data
        self.evaluation_nodes = evaluation_nodes
        self.error_density = error_density
        self.prior = prior
        self.last_evalutations = [(None, None), (None, None)]       # always keep the last two evaluations in memory

    def eval(self, z):
        if np.all(np.equal(z, self.last_evalutations[0][0])):
            pass
        elif np.all(np.equal(z, self.last_evalutations[1][0])):
            self.last_evalutations.reverse()
        else:
            self.last_evalutations[1] = (z, self.core_function(z, self.evaluation_nodes))
            self.last_evalutations.reverse()
        return self.last_evalutations[0][1]     # self.last_evalutations[0] is the most recent entry

    def posterior(self, z):
        predicted = self.eval(z)
        return self.error_density(self.data - predicted) * self.prior(z)       # implement it via mpi




