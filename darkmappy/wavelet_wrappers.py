import numpy as np
import pyssht as ssht
import pys2let as s2let


class S2letWavelets:
    """
    Linear operator wrapper for Spherical wavelet transforms using pys2let.

    All variations of the wavelet transform are supported.
    """

    def __init__(self, L, B=2, J_min=0, N=1, forward_transform=True, upsample=0):
        """Construct Spherical Wavelet transform class to hold all forward,
        inverse, and corresponding adjoint wavelet transforms.

        Args:

            L (int): Spherical harmonic bandlimit
            B (int): Wavelet tiling spread (2 = dyadic)
            J_min (int): Smallest wavelet scale
            N (int): directional samples (wigner space)
            upsample (bint): downscale wavelet scale maps.
            Reality (bint): Reality of signal.

        Raises:

            ValueError: Raised if L is not positive
            ValueError: Raised if B is less than 1.
            ValueError: Raised if N is less than 1.
            ValueError: Raised if N or B is more than L.
            WarningLog: Raised if L is very large.


        """

        if L < 1:
            raise ValueError("Bandlimit {} must be greater than 0.".format(L))

        if B < 1:
            raise ValueError("B {} must be >= 1.0".format(B))

        if N < 1:
            raise ValueError("Number of directions {} must >= 1".format(N))

        if L > 1024:
            lg.warning_log(
                "Bandlimit {} is very large, computational price is large.".format(L)
            )

        if N > L or B > L:
            raise ValueError("N and B must be >= L.")

        # Spherical Harmonic parameters
        self.L = L
        self.B = B
        self.J_min = J_min
        self.J = int(s2let.pys2let_j_max(self.B, self.L, self.J_min))
        self.N = N
        self.upsample = upsample
        self.reality = 0
        self.wav_size = self._f_wav_size()
        self.scal_size = self._f_scal_size()
        self.weights = self._wavelet_weight_map()

        # Wrap functions for optimus-primal solver
        if forward_transform:
            self.dir_op = self.forward
            self.adj_op = self.forward_adjoint
        if not forward_transform:
            self.dir_op = self.inverse
            self.adj_op = self.inverse_adjoint

    def forward(self, f, spin=0):
        """Compute the forward spherical wavelet transform.

        Args:

            f (np.complexarray): Realspace Signal
            spin (int): spin of field f

        Raises:

            ValueError: Raised if signal is nan

        """
        if f is not f:
            raise ValueError("Signal is NaN.")
        f_wav, f_scal = s2let.analysis_px2wav(
            f.flatten("C").astype(complex),
            B=self.B,
            L=self.L,
            J_min=self.J_min,
            N=self.N,
            spin=spin,
            upsample=self.upsample,
        )
        return np.concatenate((f_wav.flatten("C"), f_scal.flatten("C")), axis=0)

    def forward_adjoint(self, fws, spin=0):
        """Compute the forward_adjoint spherical wavelet transform.

        Args:

            f_ws (np.complexarray): Wavelet + scaling coefficients
            spin (int): spherical harmonic spin

        Raises:

            ValueError: Raised if wavelet coefficients nan
            ValueError: Raised if scaling coefficients nan

        """
        f_w = fws[: self.wav_size]
        f_s = fws[self.wav_size :]
        if f_s is not f_s:
            raise ValueError("Scaling Coefficients are NaN.")

        if f_w is not f_w:
            raise ValueError("Wavelet Coefficients are NaN.")

        return (
            s2let.analysis_adjoint_wav2px(
                f_w.flatten("C"),
                f_s.flatten("C"),
                B=self.B,
                L=self.L,
                J_min=self.J_min,
                N=self.N,
                spin=spin,
                upsample=self.upsample,
            )
            .flatten("C")
            .reshape((self.L, 2 * self.L - 1))
        )

    def inverse(self, fws, spin=0):
        """Compute the inverse spherical wavelet transform.

        Args:

            f_ws (np.complexarray): Wavelet + scaling coefficients
            spin (int): spherical harmonic spin

        Raises:

            ValueError: Raised if wavelet coefficients nan
            ValueError: Raised if scaling coefficients nan

        """
        f_w = fws[: self.wav_size]
        f_s = fws[self.wav_size :]
        if f_s is not f_s:
            raise ValueError("Scaling Coefficients are NaN.")

        if f_w is not f_w:
            raise ValueError("Wavelet Coefficients are NaN.")

        return (
            s2let.synthesis_wav2px(
                f_w.flatten("C"),
                f_s.flatten("C"),
                B=self.B,
                L=self.L,
                J_min=self.J_min,
                N=self.N,
                spin=spin,
                upsample=self.upsample,
            )
            .flatten("C")
            .reshape((self.L, 2 * self.L - 1))
        )

    def inverse_adjoint(self, f, spin=0):
        """Compute the inverse adjoint spherical wavelet transform.

        Args:

            f (np.complexarray): Realspace Signal
            spin (int): spin of field f

        Raises:

            ValueError: Raised if signal is nan

        """
        if f is not f:
            raise ValueError("Signal is NaN.")

        f_wav, f_scal = s2let.synthesis_adjoint_px2wav(
            f.flatten("C") + 0j,
            B=self.B,
            L=self.L,
            J_min=self.J_min,
            N=self.N,
            spin=spin,
            upsample=self.upsample,
        )
        return np.concatenate((f_wav.flatten("C"), f_scal.flatten("C")), axis=0)

    def _wavelet_weight_map(self):
        """Compute wavelet quadrature weight map"""
        offset = 0
        full_weights = self._sphere_weight_map(self.L)
        weight_map = np.ones(self.wav_size + self.scal_size)
        for j in range(self.J_min, self.J + 1):
            bandlimit = self.L
            if self.upsample == 0:
                bandlimit = min(np.ceil(self.B ** (j + 1)), self.L)
            f_size = int(bandlimit * (2 * bandlimit - 1))
            for n in range(self.N):
                weight_map[offset : offset + f_size] = self._sphere_weight_map(
                    bandlimit
                )
                offset += f_size
        return weight_map

    def _sphere_weight_map(self, L):
        """Compute spherical quadrature weight map

        Args:

                L (int): Angular bandlimit of problem
        """
        L = int(L)
        weights = np.zeros((L, 2 * L - 1))
        for row in range(L):
            theta = (row + 0.5) * np.pi / L
            weights[row, :] = np.sin(theta)
        return weights.flatten("C")

    def _f_scal_size(self):
        """Compute size of scaling coefficients."""
        bandlimit = (
            self.L
            if self.upsample == 1
            else min(np.ceil(self.B ** (self.J_min - 1)), self.L)
        )
        return int(bandlimit * (2 * bandlimit - 1))

    def _f_wav_size(self):
        """Compute size of wavelet coefficients."""
        total = 0
        for j in range(self.J_min, self.J + 1):
            bandlimit = self.L
            if self.upsample == 0:
                bandlimit = min(np.ceil(self.B ** (j + 1)), self.L)
            total += bandlimit * (2 * bandlimit - 1) * (2 * self.N - 1)
        return int(total)
