import numpy as np
import pyssht as ssht
import darkmappy.forward_models as fm


def simulate_shear_plane(
    x, sidelength, ngal=30, sigma_e=0.37, mask=None, supersample=1
):
    """Creates a realistic simulation of shear observations
    from input convergence.

    Args:

            x (np.ndarray): Input convergence map
            mask (int array): Input convergence mask map
            sidelength (int): Angular span of axes.
            ngal (float): Number density of galaxy observations to simulate.
            sigma_e (float): Intrinsic ellipticity dispersion.
    """
    super_shape = (int(x.shape[0] / supersample), int(x.shape[0] / supersample))
    model = fm.PlanarForwardModel(
        n=super_shape[0], supersample=supersample, sigma_e=sigma_e
    )
    gamma_clean = model.dir_op(kappa=x)

    gals_pix = float(ngal) * (float(sidelength) / float(super_shape[0])) ** 2

    return Gaussian_noise_plane(
        x=gamma_clean.reshape(super_shape),
        mask=mask,
        gals_pix=gals_pix,
        sigma_e=sigma_e,
    )


def Gaussian_noise_plane(x, gals_pix, sigma_e, mask=None):
    """Adds realistic noise based on observational ngal.

    Args:

            x (np.ndarray): Clean simulated data.
            mask (int array): Sky mask map
            gals_pix (float): Number of observations per pixel
            sigma_e (float): Intrinsic ellipticity dispersion.

    """
    # Output maps
    ngal_map = np.zeros(x.shape, dtype=float)
    ngal_map[:] = gals_pix
    x_noisy = np.zeros_like(x)

    A = np.sqrt(sigma_e ** 2 / (2.0 * gals_pix))
    r_vect = A * (
        np.random.randn(x.shape[0], x.shape[0])
        + 1j * np.random.randn(x.shape[0], x.shape[0])
    )
    x_noisy = x + r_vect

    if mask is not None:
        x_noisy[mask < 0.5] = 0

    return x_noisy, ngal_map


def simulate_shear_sphere(x, L, ngal=30, sigma_e=0.37, mask=None):
    """Creates a realistic simulation of shear observations
    from input convergence.

    Args:

            x (np.ndarray): Input convergence map
            mask (int array): Input convergence mask map
            L (int): Angular bandlimit.
            ngal (float): Number density of galaxy observations to simulate.
            sigma_e (float): Intrinsic ellipticity dispersion.

    Raises:

            ValueError: Input shape incorrect.
    """
    if x.shape != (L, 2 * L - 1):
        raise ValueError("Input map shape is incorrect!")

    model = fm.SphericalForwardModel(L=L, sigma_e=sigma_e)
    gamma_clean = model.dir_op(kappa=x)
    return Gaussian_noise_sphere(
        x=gamma_clean.reshape((L, 2 * L - 1)),
        mask=mask,
        L=L,
        ngal=ngal,
        sigma_e=sigma_e,
    )


def Gaussian_noise_sphere(x, L, ngal, sigma_e, mask=None):
    """Adds realistic noise based on observational ngal.

    Args:

            x (np.ndarray): Clean simulated data.
            mask (int array): Sky mask map
            L (int): Angular bandlimit.
            ngal (float): Number density of galaxy observations to simulate.
            sigma_e (float): Intrinsic ellipticity dispersion.

    Raises:

            ValueError: Input shape incorrect.

    """
    if x.shape != (L, 2 * L - 1):
        raise ValueError("Input shape is incorrect!")

    # Variance of intrinsic ellipticities distribution
    var_e = sigma_e ** 2

    # Angular conversions and factors
    area_factor = 3600.0 * (180.0 / np.pi) ** 2
    delta = 2.0 * np.pi / (2.0 * float(L) - 1.0)

    # Output maps
    ngal_map = np.zeros((L, 2 * L - 1), dtype=float)
    x_noisy = np.zeros_like(x)

    for i in range(L):
        # Compute delta theta
        theta_1 = np.pi * float(2*i+1) / ( 2.0 * float(L) - 1.0 )
        theta_2 = np.pi * float(2*i+3) / ( 2.0 * float(L) - 1.0 )
        # Compute pixel weighting factor (number density)
        c_term = 1e-5 + np.abs(np.cos(theta_1) - np.cos(theta_2))
        ngal_map[i, :] = delta * c_term * area_factor * ngal
        A = np.sqrt(var_e / (2.0 * delta * c_term * area_factor * ngal))
        # Generate a random signal with correct noise variance
        r_vect = A * (np.random.randn(2 * L - 1) + 1j * np.random.randn(2 * L - 1))
        x_noisy[i, :] = x[i, :] + r_vect

    if mask is not None:
        x_noisy[mask < 0.5] = 0

    return x_noisy, ngal_map
