import numba
import numpy as np

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