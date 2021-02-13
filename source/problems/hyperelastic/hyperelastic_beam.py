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
    """Class implementing the solution to a hyperelastic problem on a deformable beam.
    """
    def __init__(
            self, eval_times,
            n=10,
            lx=1.,
            ly=.1,
            lz=.1,
            f=(0.0, 0.0, -50.),
            nu=.3,
            time=1.,
            timestep=.1,
            tol=1e-5,
            max_iter=30,
            rel_tol=1e-10,
            log_transform=True,
            param_remapper=None):
        """Parameters
        ----------
        eval_times: numpy.ndarray
            Times at which evaluation (interpolation) of the solution is required.
        n: int, default 10
            Dimension of the grid along the smallest side of the beam
            (the grid size along all other dimensions will be scaled proportionally.
        lx: float, default 1.
            Length of the beam along the x axis.
        ly: float, default .1
            Length of the beam along the y axis.
        lz: float, default .1
            Length of the beam along the z axis.
        f: tuple or numpy.ndarray, default (0.0, 0.0, -50.)
            Force per unit volume acting on the beam.
        time: float, default 1.
            Final time of the simulation.
        timestep: float, default 1.
            Time discretization step to solve the problem.
        tol: float, default 1e-5
            Tolerance parameter to ensure the last time step is included in the solution.
        max_iter: int, default 30
            Maximum iterations for the SNES solver.
        rel_tol: int,
            Relative tolerance for the convergence of the SNES solver.
        param_remapper: object, default None
            Either None (no remapping of the parameters), or a function remapping
            the parameter of the problem (Young's modulus) to values suitable for the
            definition of the solution.
        """
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
        self.log_transform = log_transform
        self.param_remapper = param_remapper

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
        self.nu = nu
        self.ds = Measure('ds', domain=self.mesh, subdomain_data=self.boundaries)

        # evaluation times
        self.eval_times = eval_times
        self.dt = timestep
        self.T = time + tol
        self.times = np.arange(self.dt, self.T, self.dt)
        self.time = Expression('t', t=self.dt, degree=0)

    def _get_params(self, z):
        if type(z) in [list, np.ndarray]:
            param = self.param_remapper(z[0]) if self.param_remapper is not None else z[0]
        else:
            param = self.param_remapper(z) if self.param_remapper is not None else z

        e_var = variable(Constant(param))                   # Young's modulus
        nu = Constant(self.nu)                              # Shear modulus (Lamè's second parameter)
        return e_var, nu

    def _solve(self, z, x=None):
        # problem variables
        du = TrialFunction(self.V)                          # incremental displacement
        v = TestFunction(self.V)                            # test function
        u = Function(self.V)                                # displacement from previous iteration

        # kinematics
        ii = Identity(3)                                    # identity tensor dimension 3
        f = ii + grad(u)                                    # deformation gradient
        c = f.T * f                                         # right Cauchy-Green tensor

        # invariants of deformation tensors
        ic = tr(c)
        j = det(f)

        # elasticity parameters
        e_var, nu = self._get_params(z)
        mu, lmbda = e_var / (2 * (1 + nu)), e_var * nu / ((1 + nu) * (1 - 2 * nu))

        # strain energy density, total potential energy
        psi = (mu / 2) * (ic - 3) - mu * ln(j) + (lmbda / 2) * (ln(j)) ** 2
        pi = psi * dx - self.time * dot(self.f, u) * self.ds(3)

        ff = derivative(pi, u, v)                           # compute first variation of pi
        jj = derivative(ff, u, du)                          # compute jacobian of f

        # solving
        if x is not None:
            numeric_evals = np.zeros(shape=(x.shape[1], len(self.times)))
            evals = np.zeros(shape=(x.shape[1], len(self.eval_times)))
        else:
            numeric_evals = None
            evals = None
        for it, t in enumerate(self.times):
            self.time.t = t
            self.solver(ff == 0, u, self.bcs, J=jj, bcs=self.bcs, solver_parameters=self.solver_parameters)
            if x is not None:
                if self.log_transform:
                    numeric_evals[:, it] = np.log(np.array([-u(x_)[2] for x_ in x.T]).T)
                else:
                    numeric_evals[:, it] = np.array([u(x_)[2] for x_ in x.T]).T

        # time-interpolation
        if x is not None:
            for i in range(evals.shape[0]):
                evals[i, :] = np.interp(self.eval_times, self.times, numeric_evals[i, :])
        return (evals, u) if x is not None else u

    def __call__(self, z, x, reshape=True, retall=False):
        """Solves the equation for the provided parameter `z` and returns the evaluation
        for each one of the nodes in `x` and each evaluation time instant (passed at creation).

        Parameters
        ----------
        z : float or numpy.ndarray
            Value of the parameter (Young's modulus); if `z` is a vector, it has to have
            only one cell (the rest will not be considered).
        x : numpy.ndarray
            Matrix containing the coordinates of the points at which the solution has to be evaluated
            (one column per each point).
        reshape : bool, default True
            If `reshape` is set to True, then the results of the evaluations at different time points will
            be concatenated into a single vector, otherwise they are returned as a matrix whose columns
            correspond to values at different evaluation timesteps.
        retall : bool, default False
            If set to True, then also the solution at the last computed time step is returned.

        Returns
        -------
        numpy.ndarray or (numpy.ndarray, object)
            The vector or matrix of evaluation and additionally the FEniCS solution object
            (if `retall` was set to true).
        """
        evals, u = self._solve(z, x)
        evals = evals.flatten() if reshape else evals
        return (evals, u) if retall else evals

    def plot(self, z):
        """Solves the problem and then plots the beam with a colormap corresponding to the displacement.

        Parameters
        ----------
        z : float or numpy.ndarray
            Value of the parameter (Young's modulus); if `z` is a vector, it has to have
            only one cell (the rest will not be considered).
        """
        u = self._solve(z)
        vplot(u)
