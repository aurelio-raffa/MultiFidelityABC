from fenics import Expression, Constant

from source.problems.hyperelastic.hyperelastic_beam import HyperelasticBeam


class DamagedBeam(HyperelasticBeam):
    """Class implementing a damaged hyperelastic beam.
    """
    def __init__(
            self,
            eval_times,
            init_k0=5.e3,
            init_k1=4.e5,
            variable_elastic_params=False,
            **hyperelastic_kwargs):
        super().__init__(eval_times, **hyperelastic_kwargs)
        self.variable_elastic_params = variable_elastic_params
        self.init_k0 = init_k0
        self.init_k1 = init_k1

    def _get_params(self, z):
        if self.param_remapper is not None:
            params = {
                'param' + str(i): (pr(z_) if pr is not None else z_)
                for i, (pr, z_) in enumerate(zip(self.param_remapper, z))}
        else:
            params = {'param' + str(i): z_ for i, z_ in enumerate(z)}
        if not self.variable_elastic_params:
            params['param2'] = self.init_k0
            params['param3'] = self.init_k1
        tol = 1e-12
        e_var = Expression('abs(x[0] - param0) <= param1 + tol ? param2 : param3', degree=0, tol=tol, **params)
        nu = Constant(self.nu)
        return e_var, nu


