import numba
import numpy as np
import scipy as sc
import scipy.signal

@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2


def fresnel_t(beta1, beta2):
    return 2 * beta1 / (beta1 + beta2)
def fresnel_r(beta1, beta2):
    return (beta1 - beta2) / (beta1 + beta2)


def rad2deg(rad):
    return rad * 180/np.pi
def deg2rad(deg):
    return deg * np.pi /180

def normmax(X):
    """ Normalize X between Min and Max"""
    X = X-np.nanmin(X)
    return X/np.nanmax(X)

def nm2eV(nm):
    wl = nm/1e9
    c = 299792458
    nu = c/wl
    E = 4.13566769692386e-15*nu
    return E
def eV2nm(eV):
    nu = eV/4.13566769692386e-15
    c = 299792458
    wl = c/nu
    return wl*1e9


def shift_by_n(vec, n):
    """ Shift a vector by an even number of elements """
    res = np.zeros(vec.shape) * np.nan
    if n > 0:
        res[:n] = np.nan
        res[n:] = vec[:-n]
    elif n < 0:
        res[:n] = np.nan
        res[:n] = vec[-n:]
    elif n == 0:
        res = vec
    else:
        raise ValueError('Invalid n.')
    return res


def shift_by_delta(y, sft, x=None, oversampling=10, mask_extrapolated_data=True):
    """Shift a vector by any distance, with a precidion of <oversampling>
     compared to the current sampling,
     optionally on an axis x."""
    L = len(y)
    nans = np.isnan(y)
    if np.any(nans):
        y = interp_nans(y)

    if x is None:
        x = np.arange(L)
    yo, xo = sc.signal.resample(y, L * oversampling, t=x, window=('gaussian', L / 4))

    dx = np.mean(x[1:] - x[:-1])
    # print(sft,dx)
    shifto = int(np.round(oversampling * sft / dx))
    # print(sft/dx)
    yo_shifted = shift_by_n(yo, shifto)
    y_shifted = sc.signal.resample(interp_nans(yo_shifted), L)

    if mask_extrapolated_data:
        if sft < 0:
            y_shifted[int(np.floor(sft)):] = np.nan
        elif sft > 0:
            y_shifted[:int(np.ceil(sft))] = np.nan

    if any(nans):
        nans_shifted = interp_nans(shift_by_n(nans, int(np.round(shifto / oversampling))) == 1)
        y_shifted[nans_shifted] = np.nan

    return y_shifted

def interp_nans(y, modify_input = False):
    nans, x= nan_helper(y)
    if modify_input:
        y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        return
    else:
        Y = y.copy()
        Y[nans]= np.interp(x(nans), x(~nans), y[~nans])
        return Y

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]