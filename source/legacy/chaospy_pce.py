import numpoly

import chaospy as cpy
import numpy as np

from time import time
from chaospy import Normal, generate_expansion

if __name__ == '__main__':

    prior_mean = 0.
    prior_sigma = 5.
    dim = 2
    prior = cpy.Iid(Normal(prior_mean, prior_sigma), dim)

    poly_order = 2
    abscissas, weights = cpy.generate_quadrature(poly_order, prior, rule='gaussian')
    expansion = generate_expansion(poly_order, prior, retall=False)

    def forward_model(params):
        return np.prod(np.exp(-params**2))

    evals = np.array([forward_model(sample) for sample in abscissas.T])
    surrogate = cpy.fit_quadrature(expansion, abscissas, weights, evals)

    coefficients = np.array(surrogate.coefficients)
    indeterminants = surrogate.indeterminants
    exponents = surrogate.exponents

    t = time()
    for _ in range(10000):
        surrogate(.5, 1.5)
    print(time() - t)


