import numpy as np
from numpy.fft import fft2, ifft2, ifftshift, fftshift
import warnings
from enum import Enum
import darkmappy.logs as lg
import pyssht as ssht


class SphericalForwardModel:
    """
    Weak Gravitational Lensing spherical Forward model

    Supports additional complexity which should simply
    be appended to the dir_op and adj_op objects
    appropriately (e.g. psf degridding step, psf
    deconvolution etc.)
    """

    def __init__(self, L, mask=None, ngal=None, sigma_e=0.37):
        """Construct class to hold the spherical forward and
        forward adjoint operators.

        Args:

                L (int): Spherical harmonic bandlimit
                mask (int array): Map of realspace masking.
                ngal (int array): Map of galaxy observation count.
                sigma_e (float): intrinsic ellipticity dispersion

        Raises:

                ValueError: Raised if L is not positive
                ValueError: Raised if mask is the wrong shape.
                WarningLog: Raised if L is very large.
        """
        if L < 1:
            raise ValueError("Bandlimit {} must be greater than 0.".format(L))

        if L > 2048:
            lg.warning_log(
                "Bandlimit {} is very large, computational price is large.".format(L)
            )

        # General class members
        self.L = L
        self.shape = (self.L, 2 * self.L - 1)

        # Define harmonic transforms and kernel mapping
        self.harmonic_kernel = self.compute_harmonic_kernel()

        # Intrinsic ellipticity dispersion
        self.var_e = sigma_e ** 2

        # Define realspace masking
        if mask is None:
            self.mask = np.ones(self.shape, dtype=bool)
        else:
            self.mask = mask.astype(bool)

        if self.mask.shape != self.shape:
            raise ValueError("Shape of mask map is incorrect!")

        # Define observational covariance
        if ngal is None:
            self.inv_cov = self.mask_forward(np.ones(self.shape))
        else:
            self.inv_cov = self.ngal_to_inv_cov(ngal)

    def dir_op(self, kappa):
        """Spherical weak lensing measurement operator

        Args:

                kappa (complex array): Convergence signal
        """
        # 1) Compute convergence harmonic coefficients
        klm = ssht.forward(kappa, L=self.L, Spin=0)
        # 2) Map to shear harmonic coefficients
        ylm = self.harmonic_mapping(klm)
        # 3) Compute shear realspace map
        y = ssht.inverse(ylm, L=self.L, Spin=2)
        # 4) Apply observational mask
        y_obs = self.mask_forward(y)
        # 5) Covariance weight the shear observations
        return self.cov_weight(y_obs)

    def adj_op(self, gamma):
        """Spherical weak lensing adjoint measurement operator

        Args:

                gamma (complex array): Shear Observations (cov weighted)
        """
        # 1) Covariance weight the shear observations
        y_obs = self.cov_weight(gamma)
        # 2) Grid masked shear onto a full-sky
        y = self.mask_adjoint(y_obs)
        # 3) Compute shear harmonic coefficients
        ylm = ssht.inverse_adjoint(y, L=self.L, Spin=2)
        # 4) Map to convergence harmonic coefficients
        klm = self.harmonic_mapping(ylm)
        # 5) Compute convergence realspace map
        return ssht.forward_adjoint(klm, L=self.L, Spin=0)

    def sks_estimate(self, gamma):
        """Computes spherical Kaiser-Squires estimator (as a first estimate)

        Args:

                gamma (complex array): Shear Observations (full-sky)
        """
        ylm = ssht.forward(gamma, L=self.L, Spin=2)
        klm = self.harmonic_inverse_mapping(ylm)
        return ssht.inverse(klm, L=self.L, Spin=0)

    def compute_harmonic_kernel(self):
        """Compuptes harmonic space kernel mapping

        Returns:

                Harmonic space weak lensing kernel
        """
        k = np.ones(self.L ** 2, dtype=float)
        index = 4
        for l in range(2, self.L):
            for m in range(-l, l + 1):
                el = float(l)
                k[index] = -1.0 * np.sqrt(((el + 2.0) * (el - 1.0)) / ((el + 1.0) * el))
                ++index
        return k

    def harmonic_mapping(self, flm):
        """Applys harmonic space mapping.

        Args:

                flm (complex array): harmonic coefficients.

        Returns:

                Mapped harmonic coefficients glm = flm * K

        """
        out = flm * self.harmonic_kernel
        out[:4] = 0
        return out

    def harmonic_inverse_mapping(self, flm):
        """Applys harmonic space inverse mapping.

        Args:

                flm (complex array): harmonic coefficients.

        Returns:

                Inverse mapped harmonic coefficients glm = flm / K

        """
        out = flm / self.harmonic_kernel
        out[:4] = 0
        return out

    def mask_forward(self, f):
        """Applies given mask to a field.

        Args:

                f (complex array): Realspace Signal

        Raises:

                ValueError: Raised if signal is nan
                ValueError: Raised if signal is of incorrect shape.

        Returns:

                Array of observations only.

        """
        if f is not f:
            raise ValueError("Signal is NaN.")

        if f.shape != self.shape:
            raise ValueError("Signal shape is incorrect for mw-sampling")

        return f[self.mask]

    def mask_adjoint(self, x):
        """Applies given mask adjoint to observations

        Args:

                x (complex array): Set of observations.

        Raises:

                ValueError: Raised if signal is nan

        Returns:

                Gridded full-sky map of observations

        """
        if x is not x:
            raise ValueError("Signal is NaN.")

        f = np.zeros(self.shape, dtype=complex)
        f[self.mask] = x
        return f

    def ngal_to_inv_cov(self, ngal):
        """Converts galaxy number density map to data covariance.

        Assumes no correlation between pixels.

        Args:

                ngal (real array): pixel space map of number of observations per pixel

        Returns:

                Data covariance, assuming no correlations and Gaussian noise

        """
        ngal_m = self.mask_forward(ngal)
        return np.sqrt((2.0 * ngal_m) / (self.var_e))

    def cov_weight(self, x):
        """Applies covariance weighting to observations.

        Assumes no correlation between pixels.

        Args:

                x (array): pixel space map to be inverse covariance weighted.

        Returns:

                Inverse covariance weighted observations y' = y * \sigma^-1/2

        """
        return x * self.inv_cov


class PlanarForwardModel:
    """
    Weak Gravitational Lensing planar Forward model

    Supports additional complexity which should simply
    be appended to the dir_op and adj_op objects
    appropriately (e.g. psf degridding step, psf
    deconvolution etc.)
    """

    def __init__(self, n, mask=None, ngal=None, supersample=1, sigma_e=0.37):
        """Construct class to hold the spherical forward and
        forward adjoint operators.

        Args:

                n (int): Pixel count along each axis (square images)
                mask (int array): Map of realspace masking.
                ngal (int array): Map of galaxy observation count.
                supersample (float): Degree of supersampling.
                sigma_e (float): intrinsic ellipticity dispersion

        Raises:

                ValueError: Raised if map is of size 0.
                ValueError: Raised if mask is the wrong shape.
                ValueError: Raised if ngal is the wrong shape.
                WarningLog: Raised if L is very large.
        """
        if n < 1:
            raise ValueError("Input map dimensions incorrect (null dimensions)!")

        # General class members
        self.n = n
        self.shape = (n, n)
        self.super = supersample
        self.ns = int((supersample - 1) * self.n / 2)

        # Define fourier transforms and lensing kernel
        self.fourier_kernels = self.compute_fourier_kernels()

        # Intrinsic ellipticity dispersion
        self.var_e = sigma_e ** 2

        # Define realspace masking
        if mask is None:
            self.mask = np.ones(self.shape, dtype=bool)
        else:
            self.mask = mask.astype(bool)

        if self.mask.shape != self.shape:
            raise ValueError("Shape of mask map is incorrect!")

        # Define observational covariance
        if ngal is None:
            self.inv_cov = self.mask_forward(np.ones(self.shape))
        else:
            self.inv_cov = self.ngal_to_inv_cov(ngal)

    def dir_op(self, kappa):
        """Planar weak lensing measurement operator

        Args:

                kappa (complex array): Convergence signal

        """
        # 1) Compute convergence fourier coefficients
        klm = fft2(kappa)
        # 1b) Downsample map for superresolution
        if self.super > 1:
            klm = fftshift(klm)
            klm = klm[self.ns : self.n + self.ns, self.ns : self.n + self.ns]
            klm = ifftshift(klm) / self.super
        # 2) Map to shear fourier coefficients
        ylm = self.fourier_kernels[0] * klm
        # 3) Compute shear realspace map
        y = ifft2(ylm)
        # 4) Apply observational mask
        y_obs = self.mask_forward(y)
        # 5) Covariance weight the shear observations
        return self.cov_weight(y_obs)

    def adj_op(self, gamma):
        """Planar weak lensing adjoint measurement operator

        Args:

                gamma (complex array): Shear Observations (cov weighted)

        """
        # 1) Covariance weight the shear observations
        y_obs = self.cov_weight(gamma)
        # 2) Grid masked shear onto a full-sky
        y = self.mask_adjoint(y_obs)
        # 3) Compute shear fourier coefficients
        ylm = fft2(y)
        # 4) Map to convergence fourier coefficients
        klm = self.fourier_kernels[1] * ylm
        # 4b) Pad mask for superresolution
        if self.super > 1:
            klm = fftshift(klm)
            klm = np.pad(klm, ((self.ns, self.ns), (self.ns, self.ns)), "constant")
            klm = ifftshift(klm) * self.super
        # 5) Compute convergence realspace map
        return ifft2(klm)

    def ks_estimate(self, gamma):
        """Computes Kaiser-Squires estimator (as a first estimate)

        Args:

                gamma (complex array): Shear Observations (patch)
        """
        # 1) Compute shear fourier coefficients
        ylm = fft2(gamma)
        klm = np.zeros_like(ylm)
        # 2a) Map to convergence fourier coefficients
        klm[self.fourier_kernels[0] != 0] = (
            ylm[self.fourier_kernels[0] != 0]
            / self.fourier_kernels[0][self.fourier_kernels[0] != 0]
        )
        # 2b) Pad mask for superresolution
        if self.super > 1:
            klm = fftshift(klm)
            klm = np.pad(klm, ((self.ns, self.ns), (self.ns, self.ns)), "constant")
            klm = ifftshift(klm) * self.super
        # 3) Compute convergence realspace map
        return ifft2(klm)

    def compute_fourier_kernels(self):
        """Computes fourier space kernel mappings.

        Returns as a tuple {forward, inverse}.
        """
        D_f = np.zeros(self.shape, dtype=complex)

        for i in range(self.n):
            for j in range(self.n):
                kx = float(i) - float(self.n) / 2.0
                ky = float(j) - float(self.n) / 2.0
                k = kx ** 2.0 + ky ** 2.0
                if k > 0:
                    D_f[i, j] = (kx ** 2.0 - ky ** 2.0) + 1j * (2.0 * kx * ky)
                    D_f[i, j] /= k
        D_f = ifftshift(D_f)
        D_i = np.conjugate(D_f)

        return (D_f, D_i)

    def mask_forward(self, f):
        """Applies given mask to a field.

        Args:

                f (complex array): Realspace Signal

        Raises:

                ValueError: Raised if signal is nan
                ValueError: Raised if signal is of incorrect shape.

        """
        if f is not f:
            raise ValueError("Signal is NaN.")

        if f.shape != self.shape:
            raise ValueError("Signal shape is incorrect!")

        return f[self.mask]

    def mask_adjoint(self, x):
        """Applies given mask adjoint to observations

        Args:

                x (complex array): Set of observations.

        Raises:

                ValueError: Raised if signal is nan

        """
        if x is not x:
            raise ValueError("Signal is NaN.")

        f = np.zeros(self.shape, dtype=complex)
        f[self.mask] = x
        return f

    def ngal_to_inv_cov(self, ngal):
        """Converts galaxy number density map to
        data covariance.

        Assumes no intrinsic correlation between pixels.

        Args:

                ngal (real array): pixel space map of observation counts per pixel.

        """
        ngal_m = self.mask_forward(ngal)
        return np.sqrt((2.0 * ngal_m) / (self.var_e))

    def cov_weight(self, x):
        """Applies covariance weighting to observations.

        Assumes no intrinsic correlation between pixels.

        Args:

                x (array): pixel space map to be inverse covariance weighted.

        """
        return x * self.inv_cov
