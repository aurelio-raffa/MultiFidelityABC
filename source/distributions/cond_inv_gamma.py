import numpy as np
from scipy.stats import invgamma


class CondInvGamma:
    def __init__(self, alpha, beta):
        self.alpha0 = alpha
        self.beta0 = beta
        self.alpha = alpha
        self.beta = beta

    def draw(self, model, z_):
        self.alpha = self.alpha0 + len(model.data) / 2
        vec = model.data - model.eval(z_)
        self.beta = self.beta0 + (np.dot(vec, vec)) / 2
        return invgamma.rvs(self.alpha, scale=self.beta)
