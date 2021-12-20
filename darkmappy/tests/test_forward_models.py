import pytest
import numpy as np
import darkmappy.forward_models as fm
import pyssht as ssht


def test_adjoint_no_mask_spherical():
    # Define parameters
    L = 64

    # Generate random convergence
    flm = np.random.rand(L * L) + 1j * np.random.rand(L * L)
    # flm[:4] = 0  # remove mono/dipole
    k = ssht.inverse(flm, L=L)

    # Generate random shear
    flm = np.random.rand(L * L) + 1j * np.random.rand(L * L)
    flm[:4] = 0  # remove mono/dipole
    g = ssht.inverse(flm, L=L, Spin=2)

    # Generate and apply forward model
    model = fm.SphericalForwardModel(L=L)
    g = model.mask_forward(g)
    k_to_g = model.dir_op(k)
    g_to_k = model.adj_op(g)

    # Perform adjoint operator dot test.
    a = abs(np.vdot(k, g_to_k))
    b = abs(np.vdot(g, k_to_g))
    assert np.count_nonzero(k) > 0
    assert np.count_nonzero(g) > 0
    assert np.count_nonzero(k_to_g) > 0
    assert np.count_nonzero(g_to_k) > 0
    assert a == pytest.approx(b)


def test_adjoint_mask_spherical():
    # Define parameters
    L = 64
    sample_fraction = 0.5
    samples = int(sample_fraction * (L * (2 * L - 1)))

    # Generate random mask
    mask = np.zeros(L * (2 * L - 1), dtype=int)
    mask[:samples] = 1
    np.random.shuffle(mask)
    mask = mask.reshape((L, 2 * L - 1))

    # Generate random convergence
    flm = np.random.rand(L * L) + 1j * np.random.rand(L * L)
    # flm[:4] = 0  # remove mono/dipole
    k = ssht.inverse(flm, L=L)

    # Generate random shear
    flm = np.random.rand(L * L) + 1j * np.random.rand(L * L)
    flm[:4] = 0  # remove mono/dipole
    g = ssht.inverse(flm, L=L, Spin=2)

    # Generate and apply forward model
    model = fm.SphericalForwardModel(L=L, mask=mask)
    g = model.mask_forward(g)
    k_to_g = model.dir_op(k)
    g_to_k = model.adj_op(g)

    # Perform adjoint operator dot test.
    a = abs(np.vdot(k, g_to_k))
    b = abs(np.vdot(g, k_to_g))
    assert np.count_nonzero(k) > 0
    assert np.count_nonzero(g) > 0
    assert np.count_nonzero(k_to_g) > 0
    assert np.count_nonzero(g_to_k) > 0
    assert a == pytest.approx(b)


def test_adjoint_no_mask_planar():
    # Define parameters
    n = 128

    # Generate random convergence and shear
    k = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    g = np.random.randn(n, n) + 1j * np.random.randn(n, n)

    # Generate and apply forward model
    model = fm.PlanarForwardModel(n=n)
    g = model.mask_forward(g)
    k_to_g = model.dir_op(k)
    g_to_k = model.adj_op(g)

    # Perform adjoint operator dot test.
    a = abs(np.vdot(k, g_to_k))
    b = abs(np.vdot(g, k_to_g))
    assert np.count_nonzero(k) > 0
    assert np.count_nonzero(g) > 0
    assert np.count_nonzero(k_to_g) > 0
    assert np.count_nonzero(g_to_k) > 0
    assert a == pytest.approx(b)


def test_adjoint_mask_planar():
    # Define parameters
    n = 128
    sample_fraction = 0.5
    samples = int(sample_fraction * n ** 2)

    # Generate random mask
    mask = np.zeros(n * n, dtype=int)
    mask[:samples] = 1
    np.random.shuffle(mask)
    mask = mask.reshape((n, n))

    # Generate random convergence and shear
    k = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    g = np.random.randn(n, n) + 1j * np.random.randn(n, n)

    # Generate and apply forward model
    model = fm.PlanarForwardModel(n=n, mask=mask)
    g = model.mask_forward(g)
    k_to_g = model.dir_op(k)
    g_to_k = model.adj_op(g)

    # Perform adjoint operator dot test.
    a = abs(np.vdot(k, g_to_k))
    b = abs(np.vdot(g, k_to_g))
    assert np.count_nonzero(k) > 0
    assert np.count_nonzero(g) > 0
    assert np.count_nonzero(k_to_g) > 0
    assert np.count_nonzero(g_to_k) > 0
    assert a == pytest.approx(b)


def test_adjoint_superresolution_planar():
    # Define parameters
    n_in = 128
    supersample = 2
    n_out = int(n_in * supersample)

    # Generate random convergence and shear
    k = np.random.randn(n_out, n_out) + 1j * np.random.randn(n_out, n_out)
    g = np.random.randn(n_in, n_in) + 1j * np.random.randn(n_in, n_in)

    # Generate and apply forward model
    model = fm.PlanarForwardModel(n=n_in, supersample=supersample)
    g = model.mask_forward(g)
    k_to_g = model.dir_op(k)
    g_to_k = model.adj_op(g)

    # Perform adjoint operator dot test.
    a = abs(np.vdot(k, g_to_k))
    b = abs(np.vdot(g, k_to_g))
    assert np.count_nonzero(k) > 0
    assert np.count_nonzero(g) > 0
    assert np.count_nonzero(k_to_g) > 0
    assert np.count_nonzero(g_to_k) > 0
    assert a == pytest.approx(b)
