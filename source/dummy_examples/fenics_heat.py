import time
import fenics

import numpy as np
import matplotlib.pyplot as plt

from fenics import RectangleMesh, Expression, Constant, Function, Point
from fenics import FunctionSpace, TrialFunction, TestFunction, DirichletBC
from fenics import solve, lhs, rhs, errornorm, interpolate, assign
from fenics import dot, grad, dx
from fenics import File

if __name__ == '__main__':
    fenics.set_log_level(30)  # only display warnings or errors

    # here we solve a heat diffusion equation over time:
    # we use a finite difference scheme in time (backward Euler) and a variational approach in space
    # namely we iterate over (small) timesteps, each time solving a Poisson equation via finite elements

    T = 2.0             # final time
    num_steps = 50      # number of time steps
    dt = T / num_steps  # time step size

    # Create mesh and define function space
    nx = ny = 30
    mesh = RectangleMesh(Point(-2, -2), Point(2, 2), nx, ny)
    V = FunctionSpace(mesh, 'P', 1)

    # Define boundary condition
    def boundary(x, on_boundary):
        return on_boundary
    bc = DirichletBC(V, Constant(0), boundary)      # null Dirichlet conditions

    # Define initial value
    u_0 = Expression('exp(-a*pow(x[0], 2) - a*pow(x[1], 2))', degree=2, a=5)
    # the initial condition here is a "gaussian hill" of parameter alpha centered in the origin
    u_n = interpolate(u_0, V)
    # since we will be using iteratively the solution from the previous time step to compute the one
    # at the current time step, we need to convert the initial datum's expression to a Function object:
    # there are two ways to do this: either via the project() method or the interpolate() method;
    # projecy() is very popular, but since we have a closed form solution here we want to use interpolate()
    # in order to recover the exact solution within machine-error precision

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(0)
    F = u * v * dx + dt * dot(grad(u), grad(v)) * dx - (u_n + dt * f) * v * dx
    # in general, we manually define the bilinear form a(:, :) containing the unknown solution u and
    # the right-hand-side term L(:) with known terms, however this might be difficult in complicated expressions,
    # therefore we can rely on the following synthax of FEniCS that computes a and L automatically
    a, L = lhs(F), rhs(F)

    # Create VTK file for saving solution
    vtkfile = File('../../outputs/heat_gaussian/solution.pvd')
    # !!!DEVELOPER WARNING!!! change the location to save if you need to, but do not pollute the rest of repo
    # the the outputs directory is set to be ignored by git, so you won't be able to pull or push any file within!

    # this file will keep a ledger of all files generated during execution, which can be later viewed with ParaView

    # Time-stepping
    u = Function(V)
    t = 0
    for n in range(num_steps):
        t += dt                 # Update current time
        solve(a == L, u, bc)    # Compute solution
        vtkfile << (u, t)       # Save to file
        u_n.assign(u)
        # this is important as we want to keep track of the previous solution in the backward Euler schema
        # notice that we should not assign u_n = u as this would prevent us from considering u_n and u as
        # two distinct objects, which is instead what we want!
