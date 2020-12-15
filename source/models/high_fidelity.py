import numpy as np

from .base_model import ForwardModel


class HighFidelityModel(ForwardModel):
    # esempi di parametri che passo sono: forward_model, data, x, log_err_dens, log_prior;
    # perciò la fz(sol u della pde), dati (valori di u nei nodi), nodi spaziali, πe, e prior
    def __init__(self, core_function, data, evaluation_nodes, log_error_density, log_prior):
        """
        This class represents the high fidelity model that in our case coincides
        with the true forward model. We build methods to evaluate the forward
        model in a given parameter and the posterior of the parameter.
        :param core_function: ForwardModel, is our true forward model
        :param data: numpy.array, sample of observed values of the forward model
                     in the spatial nodes
        :param evaluation_nodes: numpy.array, matrix of the spatial nodes in
                                 which I have the observed data. Every column is
                                 a node
        :param log_error_density: function, is a function that requires as input
                                  a vector and returns the log density computed on
                                  it (used for computing the likelihood, passing as
                                  input the difference between the data and the
                                  vector of the evaluations of the forward model in
                                  the corresponding spatial nodes for a fixed
                                  parameter z)
        :param log_prior: function, is a function that requires in input a vector
                          (usually vector of parameters) and returns the log density
                          computed on it
        """
        # make sure that the data nodes are a subset of the quadrature nodes
        super().__init__()
        self.core_function = core_function
        self.data = data
        self.evaluation_nodes = evaluation_nodes
        self.log_error_density = log_error_density
        self.log_prior = log_prior
        # perché una valutazione è una coppia? (None,None)
        # forse perché nella coppia salvo come primo elemento il parametro z,
        # e come secondo elemento u(z) per ciascun nodo
        self.last_evaluations = [(None, None), (None, None)]  # always keep the last two evaluations in memory

    # perché confronto lo z con gli ultimi due parametri?
    # quando questa cosa è ricorrente nell'algoritmo?
    # il motivo credo sia che nel mh quando calcolo alpha al denominatore ho z_i-1
    # e non voglio ricalcolarmelo
    # ed in ogni caso perché non solo con l'ultimo dei parametri ma con gli ultimi 2?
    # forse è dovuto al fatto che se poi rigetto vuol dire che l'ultima valutaz è
    # in z* mentre la penultima è in z_i-1 e in caso di rigetto sono interessato
    # quindi alla penultima
    def eval(self, z):
        """
        Member function that returns the evaluations of the forward model for the
        parameter passed as input, in the evaluation nodes
        :param z: numpy.array, parameter for which I want to compute the values of
                  the forward model in the evaluation nodes
        :return: numpy.array, it contains the values of the forward model in the
                 evaluation nodes
        """
        if self.last_evaluations[0][0] is not None and np.all(
                np.abs(z - self.last_evaluations[0][0]) < np.finfo(np.float32).eps):
            pass
        # condition to avoid repeated evaluations for the same parameter
        elif self.last_evaluations[1][0] is not None and np.all(
                np.abs(z - self.last_evaluations[1][0]) < np.finfo(np.float32).eps):
            self.last_evaluations.reverse()
        else:
            self.last_evaluations[1] = (z, self.core_function(z, self.evaluation_nodes))
            self.last_evaluations.reverse()
        return self.last_evaluations[0][1]
        # self.last_evaluations[0] is the most recent entry

    # la posterior è prior(z)*πe(d-G(z)) ==> log(post) = log(prior) + log(πe)
    def logposterior(self, z):
        """
        Member function that returns the value of the log posterior for the parameter
        passed as input
        :param z: numpy.array, parameter for which I want to compute the log of the
                  posterior
        :return: numpy.float, value of the log of the posterior
        """
        predicted = self.eval(z)  # qui ottengo G(x_i;z)
        res = self.log_error_density(self.data - predicted) + self.log_prior(z)  # implement it via mpi
        return res
