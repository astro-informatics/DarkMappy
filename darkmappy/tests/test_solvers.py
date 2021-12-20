import pytest
import numpy as np
import darkmappy.solvers as opt


def test_l1_constrained_gaussian():
    # Define optimisation paradigm
    constrained = True

    options = {
        "tol": 1e-5,
        "iter": 5000,
        "update_iter": 50,
        "record_iters": False,
        "positivity": False,
        "real": False,
        "constrained": constrained,
        "nu": 0,
        "mu": 10.0,
    }

    PD_algo = opt.PrimalDual(data=None, phi=None, psi=None, options=options)

    # Check all parameters in this case (not in others).
    assert PD_algo.data == None
    assert PD_algo.phi == None
    assert PD_algo.psi == None
    assert PD_algo.f == None
    assert PD_algo.solver == PD_algo.l1_constrained_gaussian


def test_l1_unconstrained_gaussian():
    # Define optimisation paradigm
    constrained = False

    options = {
        "tol": 1e-5,
        "iter": 5000,
        "update_iter": 50,
        "record_iters": False,
        "positivity": False,
        "real": False,
        "constrained": constrained,
        "nu": 0,
        "mu": 10.0,
    }

    PD_algo = opt.PrimalDual(data=None, phi=None, psi=None, options=options)

    # Check all parameters in this case (not in others).
    assert PD_algo.solver == PD_algo.l1_unconstrained_gaussian
