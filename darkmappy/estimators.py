import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import warnings

import darkmappy.logs as lg
import darkmappy.forward_models as fm
import darkmappy.solvers as sv
from darkmappy.wavelet_wrappers import S2letWavelets

import optimusprimal.linear_operators as linear_operators


class DarkMappySphere:
    """
    Frontend class which handles dark matter reconstruction.

    This class currently supports the spherical lensing forward
    model, a selection of sparsity priors, for Gaussian and
    Poissonian noise models.
    """

    def __init__(
        self,
        L,
        N=1,
        data=None,
        mask=None,
        ngal=None,
        viewer=None,
        psi=None,
        constrained=True,
    ):
        """Construct class to create and hold the MAP estimator

        Args:

                L (int): Spherical harmonic bandlimit.
                N (int): Directionality of wavelet dictionary.
                data (complex array): Pixelised map of shear observations.
                mask (int array): Map of realspace masking.
                ngal (int array): Map of galaxy observation count.
                psi (linear operator): Sparsifying dictionary.
                constrained (boolean): Constrained or unconstrained optimisation

        Raises:

                ValueError: Raised if L is not positive.
                ValueError: Raised if mask is the wrong shape.
                ValueError: Raised if no data provided.
                WarningLog: Raised if no galaxy number density map provided.
                WarningLog: Raised if no masking map provided.
                WarningLog: Raised if L is very large.
                WarningLog: Raised if N is very large.
        """
        if L < 1:
            raise ValueError("Bandlimit {} must be greater than 0.".format(L))

        if L > 2048:
            lg.warning_log(
                "Bandlimit {} is very large, computational price is large.".format(L)
            )

        if N > 8:
            lg.warning_log(
                "Number of wavelet directions {} is quite large, \
				 computational price is large.".format(
                    N
                )
            )

        if data is None:
            raise ValueError("No shearing map was provided!")

        if mask is None:
            lg.info_log("No masking map was provided!")

        # General class members
        self.shape = (L, 2 * L - 1)
        self.flat = L * (2 * L - 1)

        # Define forward model & normalise
        self.phi = fm.SphericalForwardModel(L=L, mask=mask, ngal=ngal)
        self.nu = self.normalise_phi()

        if psi is None:
            self.psi = S2letWavelets(L=L, B=2, J_min=0, N=N, forward_transform=True)
        else:
            self.psi = psi

        # These simply control convergence & verbosity.
        self.options = {
            "tol": 1e-3,
            "iter": 500,
            "update_iter": 20,
            "record_iters": False,
            "positivity": False,
            "real": True,
            "constrained": constrained,
            "nu": self.nu,
            "viewer": viewer,
        }

        self.white_data = self.phi.cov_weight(self.phi.mask_forward(data))
        self.warm_start = self.phi.sks_estimate(data)

    def run_estimator(self, mu=1e-3, sigma=1):
        """Performs maximum a posteriori inference via
        proximal primal dual solver.

        Args:

                mu (float): Regularisation parameter of problem.
                sigma (float): White noise level of problem.

        """
        self.solver = sv.PrimalDual(
            data=self.white_data, phi=self.phi, psi=self.psi, options=self.options
        )
        return self.solver.solver(warm_start=self.warm_start, sigma=sigma, beta=mu)

    def normalise_phi(self):
        """Power method normalisation of forward-model"""
        r_vect = np.random.randn(self.flat) + 1j * np.random.randn(self.flat)

        return linear_operators.power_method(self.phi, r_vect.reshape(self.shape))[0]


class DarkMappyPlane:
    """
    Frontend class which handles dark matter reconstruction.

    This class currently supports the planar lensing forward
    model, a selection of sparsity priors, for Gaussian and
    Poissonian noise models.
    """

    def __init__(
        self,
        n,
        data=None,
        mask=None,
        ngal=None,
        viewer=None,
        wav=["db6"],
        levels=4,
        supersample=1,
        psi=None,
        constrained=True,
    ):
        """Construct class to create and hold the MAP estimator

        Args:

                n (int): Pixel count along each axis (square image)
                data (complex array): Pixelised map of shear observations.
                mask (int array): Map of realspace masking.
                ngal (int array): Map of galaxy observation count.
                wavs (list): Wavelet dictionaries to use.
                levels (int): Number of levels within each wavelet dictionary.
                supersample (float): Amount of supersampling.
                psi (linear operator): Sparsifying dictionary.
                constrained (boolean): Constrained or unconstrained optimisation

        Raises:

                ValueError: Raised if n is not positive.
                ValueError: Raised if mask is the wrong shape.
                ValueError: Raised if no data provided.
                WarningLog: Raised if no galaxy number density map provided.
                WarningLog: Raised if no masking map provided.
                WarningLog: Raised if n is very large.
                WarningLog: Raised if N is very large.
        """
        if n < 1:
            raise ValueError("Image size {} must be greater than 0.".format(n))

        if n > 4000:
            lg.warning_log(
                "Image size {} is very large, computational price is large.".format(n)
            )

        if data is None:
            raise ValueError("No shear map was provided!")

        if ngal is None:
            lg.warning_log("No map of observation counts was provided!")

        if mask is None:
            lg.warning_log("No masking map was provided!")

        # General class members
        self.n = n
        self.ns = int((supersample - 1) * self.n / 2)
        self.rs = ((self.ns, self.ns), (self.ns, self.ns))
        self.shape = (self.n, self.n)
        self.flat = (self.n * supersample) ** 2
        self.shape_s = (self.n * supersample, self.n * supersample)

        # Define forward model & normalise
        self.phi = fm.PlanarForwardModel(
            n=self.n, mask=mask, ngal=ngal, supersample=supersample
        )
        self.nu = self.normalise_phi()

        # These simply control convergence & verbosity.
        self.options = {
            "tol": 1e-3,
            "iter": 500,
            "update_iter": 20,
            "record_iters": False,
            "positivity": False,
            "real": False,
            "constrained": constrained,
            "nu": self.nu,
            "viewer": viewer,
        }

        self.white_data = self.phi.cov_weight(self.phi.mask_forward(data))
        self.warm_start = self.phi.ks_estimate(data)

        if psi is None:
            self.psi = linear_operators.dictionary(
                wav=wav, levels=levels, shape=self.shape_s
            )
        else:
            self.psi = psi

        self.psi.weights = np.ones(self.psi.dir_op(self.warm_start).shape)

    def run_estimator(self, mu=1e-3, sigma=1):
        """Performs maximum a posteriori inference via
        proximal primal dual solver.

        Args:

                mu (float): Regularisation parameter of problem.
                sigma (float): Whitened noise level of problem.

        """
        self.solver = sv.PrimalDual(
            data=self.white_data, phi=self.phi, psi=self.psi, options=self.options
        )
        return self.solver.solver(warm_start=self.warm_start, sigma=sigma, beta=mu)

    def normalise_phi(self):
        """Power method normalisation of forward-model"""
        r_vect = np.random.randn(self.flat) + 1j * np.random.randn(self.flat)

        return linear_operators.power_method(self.phi, r_vect.reshape(self.shape_s))[0]
