import pytest
import pyssht as ssht
import numpy as np
import darkmappy.wavelet_wrappers as op


def test_wavelet_roundtrip_error():
    # Iterate over a variety of cases
    for B in range(2, 4):
        for N in range(1, 8):
            for l in range(5, 8):

                # Define bandlimit
                L = int(2 ** l)

                # Instantiate transform class
                wavelet_ops = op.S2letWavelets(L=L, B=B, J_min=0, N=N)

                # Create Random bandlimited signal
                flm = np.random.rand(L * L) + 1j * np.random.rand(L * L)
                f = ssht.inverse(flm, L=L)
                f_ws = wavelet_ops.forward(f)
                f2 = wavelet_ops.inverse(f_ws)

                # Compute roundtrip error
                assert len(f_ws) == len(wavelet_ops.weights)
                assert f2 == pytest.approx(f)


def test_wavelet_adjoints():
    # Iterate over a variety of cases
    for B in range(2, 4):
        for N in range(1, 8):
            for l in range(5, 8):

                # Define bandlimit
                L = int(2 ** l)

                # Instantiate transform class
                wavelet_ops = op.S2letWavelets(L=L, B=B, J_min=0, N=N)

                # Create Random bandlimited signal
                flm = np.random.rand(L * L) + 1j * np.random.rand(L * L)
                f = ssht.inverse(flm, L=L)
                f_ws = wavelet_ops.forward(f)

                flm_2 = np.random.rand(L * L) + 1j * np.random.rand(L * L)
                f_2 = ssht.inverse(flm_2, L=L)
                f_ws2 = wavelet_ops.forward(f_2)

                f2_adj = wavelet_ops.forward_adjoint(f_ws2).flatten("C")

                # Compute roundtrip error
                a = abs(np.vdot(f, f2_adj))
                b = abs(np.vdot(f_ws, f_ws2))
                assert a == pytest.approx(b, 1e-1)
