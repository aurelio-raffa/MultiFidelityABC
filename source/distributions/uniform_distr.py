import numpy as np


class UnifDistr:
    def __init__(self, r, tol=1e-5):
        self.r = r
        self.tol = tol

    # q(.|z_i-1) ~ U(z_i-1-r , z_i-1+r)
    # draw verrà fatto dipendente dal parametro z dell'iterazione precedente, e sarà
    # un unif centrata nel punto precedente di raggio r, non ho ben capito il discorso di max e min
    def draw(self, z):
        # uso max e min per evitare che per un determinato r centrato
        # in un determinato z io possa uscire da [tol, 1-tol]
        lbs = np.max([z - self.r, np.zeros_like(z) - self.tol], axis=0)
        ubs = np.min([z + self.r, np.ones_like(z) + self.tol], axis=0)
        return np.random.uniform(lbs, ubs, size=z.shape)

    def logdensity(self, z1, z2):
        lbs = np.max([z2 - self.r, np.zeros_like(z2) - self.tol], axis=0)
        ubs = np.min([z2 + self.r, np.ones_like(z2) + self.tol], axis=0)
        dens = 1. / np.prod(ubs - lbs)
        return np.log(dens) if np.all(np.abs(z1 - z2) <= self.r) else -np.inf
