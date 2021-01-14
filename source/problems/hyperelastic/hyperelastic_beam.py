import dolfin
import fenics

import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
from fenics import Function, Identity, TestFunction, TrialFunction, VectorFunctionSpace
from fenics import BoxMesh, Point, CompiledSubDomain, MeshFunction, DirichletBC
from fenics import Constant, variable, derivative, Measure, Expression
from fenics import grad, det, tr, solve, dx, dot, ln, plot
from vedo.dolfin import plot as vplot

from source.utils.decorators import CountIt


class HyperelasticBeam:

    def __init__(
            self, eval_times, n=10, lx=1., ly=.1, lz=.1,
            f=(0.0, 0.0, -50.), time=1., timestep=.1,
            tol=1e-5, max_iter=30, rel_tol=1e-10, param_remappers=None):

        # solver parameters
        self.solver = CountIt(solve)
        self.solver_parameters = {
            'nonlinear_solver': 'snes',
            'snes_solver': {
                'linear_solver': 'lu',
                'line_search': 'basic',
                'maximum_iterations': max_iter,
                'relative_tolerance': rel_tol,
                'report': False,
                'error_on_nonconvergence': False}}
        self.param_remappers = param_remappers

        # mesh creation
        self.n = n
        self.lx = lx
        self.ly = ly
        self.lz = lz

        min_len = min(lx, ly, lz)
        mesh_dims = (int(n * lx / min_len), int(n * ly / min_len), int(n * lz / min_len))
        self.mesh = BoxMesh(Point(0, 0, 0), Point(lx, ly, lz), *mesh_dims)
        self.V = VectorFunctionSpace(self.mesh, 'Lagrange', 1)

        # boundary conditions
        self.left = CompiledSubDomain('near(x[0], side) && on_boundary', side=0.0)
        self.right = CompiledSubDomain('near(x[0], side) && on_boundary', side=lx)
        self.top = CompiledSubDomain('near(x[2], side) && on_boundary', side=lz)

        self.boundaries = MeshFunction('size_t', self.mesh, self.mesh.topology().dim() - 1, 0)
        self.boundaries.set_all(0)
        self.left.mark(self.boundaries, 1)
        self.right.mark(self.boundaries, 2)
        self.top.mark(self.boundaries, 3)

        self.bcs1 = DirichletBC(self.V, Constant([0.0, 0.0, 0.0]), self.boundaries, 1)
        self.bcs2 = DirichletBC(self.V, Constant([0.0, 0.0, 0.0]), self.boundaries, 2)
        self.bcs = [self.bcs1, self.bcs2]

        # surface force
        self.f = Constant(f)
        self.ds = Measure('ds', domain=self.mesh, subdomain_data=self.boundaries)

        # evaluation times
        self.eval_times = eval_times
        self.dt = timestep
        self.T = time + tol
        self.times = np.arange(self.dt, self.T, self.dt)
        self.time = Expression('t', t=self.dt, degree=0)

    def _solve(self, z, x=None):
        # problem variables
        du = TrialFunction(self.V)          # incremental displacement
        v = TestFunction(self.V)            # test function
        u = Function(self.V)                # displacement from previous iteration

        # kinematics
        ii = Identity(3)                    # identity tensor dimension 3
        f = ii + grad(u)                    # deformation gradient
        c = f.T * f                         # right Cauchy-Green tensor

        # invariants of deformation tensors
        ic = tr(c)
        j = det(f)

        # elasticity parameters
        params = deepcopy(z)
        if self.param_remappers is not None:
            for i, (fun_, z_) in enumerate(zip(self.param_remappers, z)):
                params[i] = fun_(z_) if fun_ is not None else z_

        e_var = variable(Constant(params[0]))    # Young's modulus
        nu = Constant(params[1])              # Shear modulus (Lamè's second parameter)
        mu, lmbda = e_var / (2 * (1 + nu)), e_var * nu / ((1 + nu) * (1 - 2 * nu))

        # strain energy density, total potential energy
        psi = (mu / 2) * (ic - 3) - mu * ln(j) + (lmbda / 2) * (ln(j)) ** 2
        pi = psi * dx - self.time * dot(self.f, u) * self.ds(3)

        ff = derivative(pi, u, v)         # compute first variation of pi
        jj = derivative(ff, u, du)        # compute jacobian of f

        # solving
        if x is not None:
            numeric_evals = np.zeros(shape=(*x.shape, len(self.times)))
            evals = np.zeros(shape=(*x.shape, len(self.eval_times)))
        else:
            numeric_evals = None
            evals = None
        for it, t in enumerate(self.times):
            self.time.t = t
            self.solver(ff == 0, u, self.bcs, J=jj, bcs=self.bcs, solver_parameters=self.solver_parameters)
            if x is not None:
                numeric_evals[:, :, it] = np.array([u(x_) for x_ in x.T]).T

        # time-interpolation
        if x is not None:
            for i in range(evals.shape[0]):
                for j in range(evals.shape[1]):
                    evals[i, j, :] = np.interp(self.eval_times, self.times, numeric_evals[i, j, :])
        return (evals, u) if x is not None else u

    def __call__(self, z, x, reshape=True, retall=False):
        evals, u = self._solve(z, x)
        evals = evals.flatten() if reshape else evals
        return (evals, u) if retall else evals

    def plot(self, z):
        u = self._solve(z)
        vplot(u)
