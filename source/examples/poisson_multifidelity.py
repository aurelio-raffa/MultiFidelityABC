import fenics
import time

import seaborn as sns
import pandas as pd
import numpy as np
import chaospy as cpy
import matplotlib.pyplot as plt

from copy import deepcopy
from chaospy import Normal, Uniform, generate_expansion
from fenics import UnitSquareMesh, Expression, Constant, Function
from fenics import FunctionSpace, TrialFunction, TestFunction, DirichletBC
from fenics import dot, grad, solve, plot, dx, errornorm, lhs, rhs

from source.models.high_fidelity import HighFidelityModel
from source.models.chaos_expansion import PCESurrogate
from source.core.metropolis_hastings import metropolis_hastings, adaptive_multifidelity_mh


class PoissonEquation:
    def __init__(self, grid_shape, f, init_z, dirichlet, degree=1, polynomial_type='P'):
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
        #creo finite dim function space
        # che significa polinomial_type di tipo 'P' ? ==> P lagrange polynomial
        self.V = FunctionSpace(self.mesh, polynomial_type, degree)
        self.dirichlet = DirichletBC(self.V, Expression(dirichlet, degree=degree+3), boundary)
        self._paramnames = ['param{}'.format(i) for i in range(len(init_z))]
        self.f = Expression(f, degree=degree, **dict(zip(self._paramnames, init_z)))
        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        #da qui in poi ho la riscrittura del problema in forma variazionale
        self.a = dot(grad(u), grad(v)) * dx
        self.L = self.f * v * dx
        self.u = Function(self.V)

#qui z è in generale un vettore di parametri, nel nostro esempio bidimensionale
    def _solve(self, z):
        for key, val in zip(self._paramnames, z):
            self.f.user_parameters[key] = val
        solve(self.a == self.L, self.u, self.dirichlet)

#z vettore di parametri, mentre x una matrice 2xn nel caso ho n nodi e problema bidimensionale
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


if __name__ == '__main__':
    #che cosa fa set_log_level?
    fenics.set_log_level(30)
    np.random.seed(1226)

#la G del paper sarà la soluz dell'eq di Poisson nel quadrato unitario usando griglia 32x32
#funzione f diperndente dai parametri [.5,.5], definita come l'espressione in verde,
#0 credo stia per dire condizioni omogenee di dirichlet
    forward_model = PoissonEquation(
        np.array([32, 32]), 'exp(-100*(pow(x[0] - param0, 2) + pow(x[1] - param1, 2)))',
        np.array([.5, .5]), '0')

    tol = 1e-5          # tol is used for not drawing nodes from the boundary
    num_data = 100      # sample's dimension of data collected
    noise_sigma = .75   # since we generate the data we add...
    true_z = np.array([.25, .75])
    # perché faccio l'uniforme tra [0+tol, 1-tol]
    x = np.random.uniform(0 + tol, 1 - tol, size=(2, num_data))
    true_vals = forward_model(true_z, x)
    data = true_vals + np.random.normal(0, noise_sigma, size=true_vals.shape)

#dimensione del parametro (in questo caso bidimensionale)
    dim = 2

    #dovrebbe essere f(z1,z2)=f(z1)f(z2) dove f(s) = pdf U(tol,1-tol)
    prior = cpy.Iid(Uniform(0 + tol, 1 - tol), dim)

# definisco log prior altrimenti avrei problemi di instabilità numerica
    def log_prior(z_):
        # min e max servono solo perché se passo un valore maggiore di 1-tol
        #o minore di tol
        z_ = np.min([z_, np.ones_like(z_) - tol], axis=0)
        z_ = np.max([z_, np.zeros_like(z_) + tol], axis=0)
        return np.log(prior.pdf(z_))

#questa è πe: (x1,...,xn)|--> ∏_i f(xi) con f pdf gaussiana (0, noise_sigma)
    def log_err_dens(x_):
        #ritorno la log density di gaussiana (0 ,noise_sigma) a meno di costante
        return np.sum(-x_**2/(2*noise_sigma**2))

#per l'hfm dovrò fare metropolis-hasting usando la vera posterior, proporzionale a
#prior*likelihood, dove la likelihood è il prodotto delle err_dens nei punti d_i-G(x_i;z)
#perciò ad ogni passo del mh dovrò calcolare per il parametro z proposto la soluzione
#della pde nei vari nodi x_i (tipicamente operazione costosa la risoluzione della pde).
    #passerò come parametri la funzione (intesa come la soluzione della pde), i dati
    #che serviranno per il calcolo della likelihood, i nodi su cui andrò a valutare la funzione
    #error_dens che mi servirà per calcolo della likelihood, e prior che servirà per calcolare
    #la posterior (queste ultime come log sempre per motivi di instabilità numerica)
    hfm = HighFidelityModel(
        forward_model, data, x, log_err_dens, log_prior)

#classe proposal density, che servirà nel mh, in questo caso uniforme con raggio definito
#
    class UnifProposal:
        def __init__(self, r):
            self.r = r

# q(.|z_i-1) ~ U(z_i-1-r , z_i-1+r)
#draw verrà fatto dipendente dal parametro z dell'iterazione precedente, e sarà
#un unif centrata nel punto precedente di raggio r, non ho ben capito il discorso di max e min
        def draw(self, z):
            #uso max e min per evitare che per un determinato r centrato
            #in un determinato z io possa uscire da [tol, 1-tol]
            lbs = np.max([z - self.r, np.zeros_like(z) - tol], axis=0)
            ubs = np.min([z + self.r, np.ones_like(z) + tol], axis=0)
            return np.random.uniform(lbs, ubs, size=z.shape)


        def logdensity(self, z1, z2):
            lbs = np.max([z2 - self.r, np.zeros_like(z2) - tol], axis=0)
            ubs = np.min([z2 + self.r, np.ones_like(z2) + tol], axis=0)
            dens = 1./np.prod(ubs-lbs)
            return np.log(dens) if np.all(np.abs(z1 - z2) <= self.r) else -np.inf

#definisco il low fidelity model che sarà un'approssimazione poolinomiale, della funzione vera
#(soluzione della pde), che stimo a partire dai dati.
#non mi è chiaro perché passo sia prior sia log_prior
#poi secifico il grado dei polinomi approssimanti ed il multi_fidelity_q
#PCESurrogate fa la stessa cosa cosa che fa la classe LowFidelityModel?
    lfm = PCESurrogate(data, log_err_dens, prior, log_prior, 2, 10)
    lfm.fit(hfm)

    proposal = UnifProposal(.05)
    samples = 10000
    init_z = np.array([.5, .5])

#questo è l'm del paper, ossia il numero di iterazioni che faccio all'interno del mh dentro
#il multifidelity
    subchain_len = 100
    upper_th = 1e-4
    error_th = 1e-2
    init_radius = .1
    rho = .9
    mfm = deepcopy(lfm)

    lfmh_t = time.time()
    lfmh = metropolis_hastings(lfm, proposal, init_z, samples)
    lfmh_t = time.time() - lfmh_t

    hfmh_t = time.time()
    hfmh = metropolis_hastings(hfm, proposal, init_z, samples)
    hfmh_t = time.time() - hfmh_t

    mfmh_t = time.time()
    #quante sono nel multifidelity il numero di z estratte alle fine? sempre samples?
    #sembrerebbe di si
    mfmh = adaptive_multifidelity_mh(
        subchain_len, samples // subchain_len, upper_th, error_th, init_radius, rho, mfm, hfm, proposal, init_z)
    mfmh_t = time.time() - mfmh_t

    print('\nperformance evaluation:')
    print('\tlow fidelity:\t{:.2f}s ({:.4}s per iteration)'.format(lfmh_t, lfmh_t/samples))
    print('\thigh fidelity:\t{:.2f}s ({:.4}s per iteration)'.format(hfmh_t, hfmh_t/samples))
    print('\tmulti fidelity:\t{:.2f}s ({:.4}s per iteration)'.format(mfmh_t, mfmh_t/samples))

    plt.figure()

    for i in range(2):
        plt.subplot(2, 2, i+1)
        plt.plot(hfmh[i, :], label='true model MH')
        plt.plot(lfmh[i, :], label='low-fidelity model MH')
        plt.plot(mfmh[i, :], label='adaptive MH')
        plt.legend()

    burn = 400
    for i in range(2):
        plt.subplot(2, 2, i+3)

        mh_data = pd.DataFrame({
            'data': np.concatenate([hfmh[i, burn:], lfmh[i, burn:], mfmh[i, burn:]], axis=0),
            'method':
                ['true model MH samples'] * (samples - burn) +
                ['low-fidelity model MH samples'] * (samples - burn) +
                ['adaptive MH samples'] * (samples - burn)})
        sns.histplot(
            mh_data,
            x='data',
            hue='method',
            bins=50,
            multiple='layer',
            edgecolor='.3',
            linewidth=.5)

    plt.show()

# guardando i plot perché gli istogrammi del true model e dell'adaptive non sono centrati
# nei true value dei parametri?