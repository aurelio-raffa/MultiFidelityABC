import numpy as np
import matplotlib.pyplot as plt

from fenics import UnitSquareMesh, Expression, Function
from fenics import FunctionSpace, TrialFunction, TestFunction, DirichletBC
from fenics import dot, grad, solve, dx
from scipy.stats import logistic

from source.utils.decorators import CountIt


class PoissonEquation:
    """Class implementing a linear, 2D Poisson equation on the unit square
    with Dirichlet boundary conditions. Provides methods to compute a numerical solution
    (through FEniCS) on a set of spatial points and plot its contour.
    """
    def __init__(self, grid_shape, f, init_z, dirichlet, degree=1, polynomial_type='P', reparam=True):
        """Parameters
        ----------
        grid_shape : numpy.array or list
            Defines the grid dimensions of the mesh used to solve the problem.
        f : str
            Source term of the Poisson equation in a form accepted by FEniCS (C++ style string)
        init_z : numpy.ndarray
            Placeholder value(s) for parameters of the model.
        dirichlet : str
            Dirichlet boundary conditions in string form accepted by FEniCS.
        degree : int, default 1
            Polynomial degree for the functional space.
        polynomial_type : str, default 'P'
            String encoding the type of polynomials in the functional space, according to FEniCS conventions
            (defaults to Lagrange polynomials).
        reparam: bool, default True
            Boolean indicating whether input parameters are to be reparametrized according to
            an inverse-logit transform.
        """
        def boundary(x, on_boundary):
            return on_boundary

        self.grid_shape = grid_shape
        self.mesh = UnitSquareMesh(*grid_shape)

        self.V = FunctionSpace(self.mesh, polynomial_type, degree)
        self.dirichlet = DirichletBC(self.V, Expression(dirichlet, degree=degree + 3), boundary)
        self._paramnames = ['param{}'.format(i) for i in range(len(init_z))]
        self.f = Expression(f, degree=degree, **dict(zip(self._paramnames, init_z)))
        u = TrialFunction(self.V)
        v = TestFunction(self.V)

        self.a = dot(grad(u), grad(v)) * dx
        self.L = self.f * v * dx
        self.u = Function(self.V)
        self.reparam = reparam
        self.solver = CountIt(solve)

    def _solve(self, z):
        if self.reparam:
            z = logistic.cdf(z)
        for key, val in zip(self._paramnames, z):
            self.f.user_parameters[key] = val
        self.solver(self.a == self.L, self.u, self.dirichlet)

    def __call__(self, z, x):
        """Returns the logarithm of the solution to the Poisson equation for the parameters `z`
        in the spatial nodes `x`.

        Parameters
        ----------
        z : numpy.array
            Parameter values for the equation.
        x : numpy.array
            Matrix whose columns are the spatial nodes.

        Returns
        -------
        numpy.ndarray
            Logarithm of the solution of the Poisson equation.
        """
        self._solve(z)
        return np.array([np.log(self.u(x_)) for x_ in x.T])

    def plot(self, z):
        """Plots the logarithm of the solution to the Poisson equation for the parameters `z`
        on the domain (unit square).

        Parameters
        ----------
        z : numpy.array
            Parameter values for the equation.
        """
        self._solve(z)
        plt.figure()
        plt.contourf(
            np.log(self.u.compute_vertex_values(self.mesh).reshape(*(self.grid_shape + 1))),
            cmap='viridis')
        plt.show()
