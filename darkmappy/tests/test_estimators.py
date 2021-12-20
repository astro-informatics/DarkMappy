import pytest
import numpy as np
import darkmappy.estimators as dm
from darkmappy.wavelet_wrappers import S2letWavelets
import optimusprimal.linear_operators as linear_operators


def test_spherical_darkmappy_instantiation():

    L = 128
    N = 20
    data = np.random.randn(L, 2 * L - 1) + 1j * np.random.randn(L, 2 * L - 1)
    estimator = dm.DarkMappySphere(L=128, data=data)


def test_planar_darkmappy_instantiation():

    n = 1024
    data = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    estimator = dm.DarkMappyPlane(n=n, data=data)
