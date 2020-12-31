import numpy as np
import matplotlib.pyplot as plt

from fenics import UnitSquareMesh, Expression, Function
from fenics import FunctionSpace, TrialFunction, TestFunction, DirichletBC
from fenics import dot, grad, solve, dx
from scipy.stats import logistic


class PoissonEquation:
    def __init__(self, grid_shape, f, init_z, dirichlet, degree=1, polynomial_type='P', reparam=True):
        """
        This class represents the Poisson equation and an object of this class can be
        used as callable object in order to obtain the value of the solution in the
        spatial nodes of interest. It provides also a method for displaying the contour
        plot.
        :param grid_shape: numpy.array, it is used to define the mesh used by fenics
                           library for approximating the solution of the pde
        :param f: str, it is the expression of the right member of the Poisson equation
        :param init_z: numpy.array, it represents the parameter (in general multidimensional)
                       from which the equation depends
        :param dirichlet: str, it is the expression for the boundary conditions
        :param degree: int, it specifies the degree of the polynomials in the
                       function space
        :param polynomial_type: str, it specifies the type of the polynomials
                                in the function space
        """

        def boundary(x, on_boundary):
            return on_boundary

        self.grid_shape = grid_shape
        self.mesh = UnitSquareMesh(*grid_shape)
        # creo finite dim function space
        # che significa polinomial_type di tipo 'P' ? ==> P lagrange polynomial
        self.V = FunctionSpace(self.mesh, polynomial_type, degree)
        self.dirichlet = DirichletBC(self.V, Expression(dirichlet, degree=degree + 3), boundary)
        self._paramnames = ['param{}'.format(i) for i in range(len(init_z))]
        self.f = Expression(f, degree=degree, **dict(zip(self._paramnames, init_z)))
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        # da qui in poi ho la riscrittura del problema in forma variazionale
        self.a = dot(grad(u), grad(v)) * dx
        self.L = self.f * v * dx
        self.u = Function(self.V)
        self.reparam = reparam

    # qui z è in generale un vettore di parametri, nel nostro esempio bidimensionale
    def _solve(self, z):
        if self.reparam:                                    # MODIFIED
            z = logistic.cdf(z)
        for key, val in zip(self._paramnames, z):
            self.f.user_parameters[key] = val
        solve(self.a == self.L, self.u, self.dirichlet)

    # z vettore di parametri, mentre x una matrice 2xn nel caso ho n nodi e problema bidimensionale
    def __call__(self, z, x):
        """
        returns the log of the solution of the Poisson equation for the parameters z
        in the spatial nodes x
        :param z: numpy.array, parameter values for the equation
        :param x: numpy.array, matrix whose columns are the spatial nodes
        :return: numpy.array, log of the solution of the Poisson equation
        """
        self._solve(z)
        return np.array([np.log(self.u(x_)) for x_ in x.T])

    def plot(self, z):
        self._solve(z)
        plt.figure()
        plt.contourf(
            np.log(self.u.compute_vertex_values(self.mesh).reshape(*(self.grid_shape + 1))),
            cmap='viridis')
        plt.show()
