import fenics

import numpy as np
import matplotlib.pyplot as plt

from fenics import UnitSquareMesh, Expression, Constant, Function
from fenics import FunctionSpace, TrialFunction, TestFunction, DirichletBC
from fenics import dot, grad, solve, plot, dx, errornorm

if __name__ == '__main__':
    fenics.set_log_level(30)        # only display warnings or errors

    # Create mesh and define function space
    grid_shape = np.array([32, 32])
    poly_degree = 1
    mesh = UnitSquareMesh(*grid_shape)
    V = FunctionSpace(mesh, 'P', poly_degree)
    # creates a grid_shape[0+1 x grid_shape[1]+1 vertices mesh with Lagrange polynomials of order order_degree

    # Define boundary condition
    u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=poly_degree+3)
    # the expression accepts a string with commands in C++ style, that is then compiled for efficiency
    # the degree parameter specifies the polynomial degree for interpolation of the solution
    # if using the exact solution, the order should be at least a few units more than the functional space's elements

    def boundary(x, on_boundary):
        return on_boundary
    bc = DirichletBC(V, u_D, boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(-6.0)
    # we can using an inline if test:
    # this method works only for simple shaped subdomains that can be expressed in terms of geometric inequalities
    tol = 1E-14
    k_0 = 1.0
    k_1 = 0.01
    c_0 = 0.3
    c_1 = 0.1
    r = 0.2
    kappa = Expression('pow(x[0] - c_0,2) + pow(x[1] - c_1,2) <= pow(r,2)  + tol ? k_1 : k_0',
                       degree=0, c_0=c_0, c_1=c_1, r=r, tol=tol, k_0=k_0, k_1=k_1)
    a = kappa*dot(grad(u), grad(v))*dx
    L = f*v*dx

    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)
    # since u is created as a Function object, it can be evaluated at any new point - although
    # this is an expensive operation!

    # Compute error in L2 norm
    error_L2 = errornorm(u_D, u, 'L2')

    # Compute maximum error at vertices
    vertex_values_u_D = u_D.compute_vertex_values(mesh)
    vertex_values_u = u.compute_vertex_values(mesh)

    # plot solution
    plt.figure()
    plt.contourf(vertex_values_u.reshape(*(grid_shape + 1)), cmap='viridis')
    plt.show()

    error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))
    # Print errors
    print('error_L2 =', error_L2)
    print('error_max =', error_max)
