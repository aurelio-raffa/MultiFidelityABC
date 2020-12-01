import pde
import time

import numpy as np

if __name__ == '__main__':

    class DiffPDE:
        def __init__(self, gridscale_x, gridscale_y, init_state_expression, t):
            self.grid = pde.UnitGrid([gridscale_x, gridscale_y])
            self.scale = np.array([gridscale_x, gridscale_y])
            self.t = t
            self.init_state = pde.ScalarField.from_expression(self.grid, init_state_expression)

        def plot_init_state(self):
            self.init_state.plot()

        def eval(self, z):
            eq = pde.DiffusionPDE(diffusivity=z)
            result = eq.solve(self.init_state, t_range=self.t)
            return result

        def __call__(self, z, x):
            nd = x.shape[1]
            if type(z) in [int, float, np.int, np.float]:
                n_evals = 1
                z = np.array([z])
            else:
                n_evals = len(z)
            return_mat = np.zeros((n_evals, nd))
            for i, z_ in enumerate(z):
                result = self.eval(z_)
                for j in range(x.shape[1]):
                    return_mat[i, j] = result.interpolate(x[:, j])
            return return_mat

    diff_eq = DiffPDE(32, 32, 'sin(x/5) * cos(y/5) * exp(-((x-50)/30)**2 -((y-50)/30)**2) * 100', 10)

    zs = np.array([1., 2., 3.])
    xs = np.random.uniform(0, 1, size=(2, 20))

    print(diff_eq(zs, xs))