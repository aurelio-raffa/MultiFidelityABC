import chaospy as cpy
import numpy as np
import progressbar as pb

from chaospy import Normal, generate_expansion

from source.models.surrogate import SurrogateModel
from source.utils.decorators import time_it


class PCESurrogate(SurrogateModel):
    # es. di parametri passati: data, log_err_dens, prior, log_prior, 2, 10
    # da capire meglio cosa sono i multi_fidelity_q
    # degree dovrebbe essere il grado dei polinomi phi_n che uso in polyn chaos exp

    # mentre multi_fidelity_q dovrebbe essere il num di pti che estraggo quando faccio
    # l'update del low-fidelity quando uso l'adaptive multi_fidelity
    def __init__(self, data, log_error_density, prior, log_prior, degree, multi_fidelity_q):
        """
        This class represents the low-fidelity model, approximating the forward
        model through polynomial chaos expansion. It has methods to fit the
        low-fidelity model, to evaluate it in a given parameter, to compute
        the posterior (based on low-fidelity) of a parameter and a method to
        update the model (called by the multi fidelity algorithm).
        :param data: numpy.array, sample of observed values of the forward model
                     in the spatial nodes
        :param log_error_density: function, it computes the log density at the input
                                  vector (used for computing the likelihood, passing
                                  as input the difference between the data and the
                                  vector of the evaluations of the surrogate model
                                  in the corresponding spatial nodes for a fixed
                                  parameter z)
        :param prior: chaospy.Iid, object that represents the prior of the parameter
        :param log_prior: function, it computes the log prior at the input vector
                          (usually vector of parameters)
        :param degree: int, it indicates the degree of the polynomial expansions
        :param multi_fidelity_q: int, it indicates the number of drawn parameter
                                 samples when I update the low fidelity model
        """
        super().__init__(data, log_error_density, prior, log_prior, multi_fidelity_q)
        self.degree = degree
        self.expansion = None
        self.proxy = None

    @staticmethod
    def _get_coeffs(fitted_poly):       # fitted_poly è la somma dei polinomi phi_n
        return np.concatenate([np.array(p.coefficients).reshape(1, -1) for p in fitted_poly], axis=0)

    # da capire meglio cos'è quadrature_rule
    # dovrebbe essere: ∫ phi_i(z)phi_j(z) f_Z(z) dz= 0 (ortogonalità), con f_Z= prior
    # e con quadrature_rule='gaussian' cosa intendo?

    @time_it(only_time=True)
    def fit(self, high_fidelity, num_evals=None, quadrature_rule='gaussian'):
        """
        Method for fitting the low-fidelity model basing on polynomial chaos
        expansion.
        :param num_evals:
        :param high_fidelity: HighFidelityModel, model that we want to approximate
                              through the low-fidelity
        :param quadrature_rule: str, rule used for the quadrature
        :return: none
        """
        # come faccio a sapere il Q (numero di punti per la stima dei coeff) del paper, cioè
        # la lunghezza di abscissae? lo fa da solo sulla base del grado? perché devo avere
        # Q>numero di polinomi

        abscissae, weights = cpy.generate_quadrature(self.degree, self.prior, rule=quadrature_rule)
        self.expansion = generate_expansion(self.degree, self.prior, retall=False)
        widgets = ['fit\t', pb.Percentage(), ' ', pb.Bar('='), ' ', pb.AdaptiveETA(), ' - ', pb.Timer()]
        bar = pb.ProgressBar(maxval=abscissae.T.shape[0], widgets=widgets)
        evals = []
        bar.start()
        for i, z_ in enumerate(abscissae.T):
            evals.append(high_fidelity.eval(z_))
            bar.update(i + 1)
        # come entrano i weights nel problema di minimizzazione?
        # cioè a questo punto ho z1,...,zQ; w1,...,wQ e G(z1),...., G(zQ)
        # secondo me entrano nella funzione di minimizzazione, dando più peso
        # agli errori commessi negli zj con peso maggiore
        self.proxy = cpy.fit_quadrature(self.expansion, abscissae, weights, evals)
        self._fit = True

    def _eval_subroutine(self, z):
        return self.proxy(*z)

    def multi_fidelity_update(self, y, radius, high_fidelity):
        """
        Member function that updates the low-fidelity, based on the polynomial chaos
        expansion of the difference between the high and low fidelity models.
        It uses a regression method on new points drawn from a uniform
        distribution centered on y with radius passed as input
        :param y: float, value of the center used for drawing the sample
        :param radius: float, value of the radius used for drawing the sample
        :param high_fidelity: HighFidelityModel, model that we want to approximate
                              with the low-fidelity
        :return: none
        """
        new_points = super().multi_fidelity_update(y, radius, high_fidelity)
        # questo è nel paper C(z_)_i per ogni nodo spaziale i per ogni z_ estratto dalla palla
        # centrata in y
        new_evals = [high_fidelity.eval(z_) - self.eval(z_) for z_ in new_points.T]
        # perché qui uso fit_regression anziché fit_quadrature per fare il pce
        # per l'errore ?
        new_poly = cpy.fit_regression(self.expansion, new_points, new_evals)
        self.proxy = self.proxy + new_poly
