import numba

@numba.vectorize([numba.float64(numba.complex128),numba.float32(numba.complex64)])
def abs2(x):
    return x.real**2 + x.imag**2


def fresnel_t(beta1, beta2):
    return 2 * beta1 / (beta1 + beta2)
def fresnel_r(beta1, beta2):
    return (beta1 - beta2) / (beta1 + beta2)