# MultiFidelityABC


## Purpose

This project was created for the 2020 edition of the [Bayesian Statistics Course](https://www4.ceda.polimi.it/manifesti/manifesti/controller/ManifestoPublic.do?EVN_DETTAGLIO_RIGA_MANIFESTO=EVENTO&c_insegn=097659&aa=2017&k_cf=225&k_corso_la=487&ac_ins=0&k_indir=MST&lang=EN&tipoCorso=ALL_TIPO_CORSO&semestre=1&idItemOfferta=132438&idRiga=219247&codDescr=097659)
for [Mathematical Engineering](https://www.mate.polimi.it/im/index.php?settore=magistrale&id_link=97&) ([Politecnico di Milano](https://www.polimi.it/en/)).

The goal was to implement and test an Adaptive Multi-Fidelity algorithm for Monte Carlo Markov Chain (MCMC) simulations
on Bayesian Inverse Problems, as proposed by [Liang Yan and Tao Zhou](https://www.sciencedirect.com/science/article/pii/S0021999119300063)
in the Journal of Computational Physics (volume 381, 15 March 2019, pages 110-128, also available [here](https://arxiv.org/abs/1807.00618)).

<br>


## Environment Setup

We use [Anaconda](https://www.anaconda.com/products/individual) to manage our execution environments.
Once you have Anaconda installed, 

* if your operative system is OSX (64-bit), please open a terminal folder at the `/requirements/` directory of this 
repository and run
    ```bash
    conda create --name myenv --file requirements_osx-64.txt
    ```

* if instead your operative system is Ubuntu, you can navigate to the same folder and execute the following command:
    ```bash
    conda create --name myenv --file requirements_ubuntu-64.txt
    ```

unfortunately as of today it is not possible to install the `fenics` package on Windows OS, 
so please in this case install [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/install-win10)
and then refer to the procedure for Ubuntu. More information can be found on the website of the [FEniCS Project](https://fenicsproject.org).

<br>


## Examples

We have implemented two test cases:
* `/source/examples/poisson_multifidelity.py` (inspired by [this FEniCS tutorial](https://fenicsproject.org/pub/tutorial/sphinx1/._ftut1003.html#mathematical-problem-formulation)), 
    where we were able to obtain the following results
    ```text
    true model (MH):
        wall time: .................................. 335.03s
        (avg.) time per iteration: .................. 0.0067s
        effective sample size (min.): ............... 2928 / 50000
        true model eval.: ........................... 50001
    ────────────────────────────────────────────────────────────
    PCE surr. (adap. MH):
        wall time: .................................. 85.62s
        (avg.) time per iteration: .................. 0.0017s
        effective sample size (min.): ............... 2953 / 50000
        true model eval.: ........................... 126
        fitting time: ............................... 0.1264s
        true mod. eval. during fitting: ............. 9
    ────────────────────────────────────────────────────────────
    GPR surr. (adap. MH):
        wall time: .................................. 29.94s
        (avg.) time per iteration: .................. 0.0006s
        effective sample size (min.): ............... 3042 / 50000
        true model eval.: ........................... 113
        fitting time: ............................... 0.1576s
        true mod. eval. during fitting: ............. 20
    ```
    namely a speedup of a factor 3.9 for the surrogate proposed in the paper versus a speedup factor of more than 11
    with our own gaussian process surrogate;
* `/source/examples/hyperelastic_multifidelity.py` (inspired by the example in [this paper](https://www.sciencedirect.com/science/article/pii/S0307904X18302063)),
    where we obtained
    ```text
    true model (MH):
        wall time: .................................. 21818.11s
        (avg.) time per iteration: .................. 1.4545s
        effective sample size (min.): ............... 590 / 15000
        true model eval.: ........................... 60004
    ────────────────────────────────────────────────────────────
    PCE surr. (adap. MH):
        wall time: .................................. 117.08s
        (avg.) time per iteration: .................. 0.0078s
        effective sample size (min.): ............... 608 / 15000
        true model eval.: ........................... 240
        fitting time: ............................... 20.0727s
        true mod. eval. during fitting: ............. 64
    ────────────────────────────────────────────────────────────
    GPR surr. (adap. MH):
        wall time: .................................. 98.89s
        (avg.) time per iteration: .................. 0.0066s
        effective sample size (min.): ............... 440 / 15000
        true model eval.: ........................... 240
        fitting time: ............................... 61.6650s
        true mod. eval. during fitting: ............. 200
    ```
    i.e. a speedup of roughly 186 times for the surrogate presented by the authors versus a speedup of 220 times for our 
    gaussian process surrogate;
  
the first example is extensively documented in the relative 
Jupyter notebook (you can find it [here](https://github.com/aurelio-raffa/MultiFidelityABC/blob/main/source/examples/poisson_multifidelity.ipynb)); \
for the second example we opted for a more traditional script, since its 
runtime is around 6 hours on a laptop and it leaves little space for interactivity (if you're feeling bored while 
running it and your device allows for interactive plotting, check out [this](https://github.com/aurelio-raffa/MultiFidelityABC/blob/main/source/utils/realtime.py) 
script enabling real-time traceplots!).

<br>


## References

* L. Yan and T. Zhou. [_Adaptive multi-fidelity polynomial chaos approach to bayesian inference in inverse problems_](https://www.sciencedirect.com/science/article/pii/S0021999119300063). 
Journal of Computational Physics, Volume 381, 15 March 2019, Pages 110-128, 2018.
* H. P. Langtangen and A. Logg. [_Solving PDEs in Python - The FEniCS Tutorial I_](https://fenicsproject.org/pub/tutorial/html/ftut1.html). 
  Springer, 2016.
* M. Hadigola and A. Doostan.[ _Least squares polynomial chaos expansion: A review of sampling strategies_](https://arxiv.org/abs/1706.07564). 
  Computer Methods in Applied Mechanics and Engineering, Volume 332, 2017.
* P. Hauseux, J. S. Hale, S. Cotin, S. P.A. Bordas 
  [_Quantifying the uncertainty in a hyperelastic soft tissue model with stochastic parameters_](https://www.sciencedirect.com/science/article/abs/pii/S0307904X18302063). 
  Applied Mathematical Modelling, Volume 62, Pages 86-102, 2018.