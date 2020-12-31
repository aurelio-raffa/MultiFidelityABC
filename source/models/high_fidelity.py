import numpy as np

from source.models.base_model import ForwardModel


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
        super().__init__(data, log_error_density, log_prior)
        self.core_function = core_function
        self.evaluation_nodes = evaluation_nodes

    # perché confronto lo z con gli ultimi due parametri?
    # quando questa cosa è ricorrente nell'algoritmo?
    # il motivo credo sia che nel mh quando calcolo alpha al denominatore ho z_i-1
    # e non voglio ricalcolarmelo
    # ed in ogni caso perché non solo con l'ultimo dei parametri ma con gli ultimi 2?
    # forse è dovuto al fatto che se poi rigetto vuol dire che l'ultima valutaz è
    # in z* mentre la penultima è in z_i-1 e in caso di rigetto sono interessato
    # quindi alla penultima
    def _eval_subroutine(self, z):
        return self.core_function(z, self.evaluation_nodes)
