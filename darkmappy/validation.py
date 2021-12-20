import numpy as np
import darkmappy.logs as lg


def analyse(x, y):
    """Computes statistics on recovered result

    Args:

            x (complex array): True map
            y (complex array): Estimated map
    """
    return snr(x, y), p_correlation(x, y)


def snr(x, y):
    """Computes signal to noise ratio in dB


    Args:

            x (np.ndarray): True map
            y (np.ndarray): Estimated map

    Raises:

            ValueError: Raised if x and y not of the same rank and size
    """
    if x.size != y.size:
        raise ValueError("Validation (SNR): Input signal shapes do not agree!")
    l2_true = np.linalg.norm(x, ord=2)
    l2_diff = np.linalg.norm(x - y, ord=2)
    return 10.0 * np.log(l2_true / l2_diff)


def p_correlation(x, y):
    """Computes pearson correlation coefficient.


    Args:

            x (np.ndarray): True map
            y (np.ndarray): Estimated map

    Raises:

            ValueError: Raised if x and y not of the same rank and size
    """
    if x.size != y.size:
        raise ValueError(
            "Validation (P-correlation): Input signal shapes do not agree!"
        )
    xr = np.real(x)
    yr = np.real(y)
    numerator = np.sum((xr - np.nanmean(xr)) * (yr - np.nanmean(yr)))
    x_sq = np.sum((xr - np.nanmean(xr)) ** 2)
    y_sq = np.sum((yr - np.nanmean(yr)) ** 2)
    return numerator / np.sqrt(x_sq * y_sq)
