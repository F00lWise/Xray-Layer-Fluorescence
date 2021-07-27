from xlfluor import Element
from xlfluor import Composite
from FLASHutil import little_helpers as lh
import scipy.constants as constants
C_r0  = constants.physical_constants['classical electron radius'][0]
import numpy as np

def fresnel_t(beta1, beta2):
    return 2 * beta1 / (beta1 + beta2)
def fresnel_r(beta1, beta2):
    return (beta1 - beta2) / (beta1 + beta2)


class Layer:
    """
    Container class for layer data
    """

    def __init__(self, element, thickness, depth_resolution=None, inelastic_cross=0, density=None, sigma=None):

        if density is not None:  # Specifit density specified, so I need to re-define
            if type(element) is Element:
                element = Element(name=element.name, Z=element.Z, atom_weight=element.atom_weight, density=density)
            elif type(element) is Composite:
                element = Composite(name=element.name, elements=element.elements, density=density,
                                    composition=element.composition)
        self.el = element
        self.d = thickness * 1e-9  # nm->m
        self.d_steps = depth_resolution  # Number of steps in which the thickness is divided for calculations
        self.sigma_inel = inelastic_cross
        self.sigma = None if sigma is None else sigma * 1e-9  # surface roughness

    def beta(self, E, theta):
        wavl_given = lh.eV2nm(E) * 1e-9  # in meters
        prefactor = self.el.atomar_density * wavl_given ** 2 / (2 * np.pi)
        f = self.el.f(E)
        self.cdelta = prefactor * C_r0 * (-np.real(f) + 1j * np.imag(f))
        return np.sqrt(1 + (2 * self.cdelta / theta ** 2))

    def L(self, E, theta, d=None):
        """
        Calculate transfer matrix
        :param E: Photon Energy
        :param theta: grazing angle
        :param d: thickness
        :return: L (transfer matrix)
        """
        # This Matrix assumes that L20@L01 = L21, which is only the case when surface roughness sigma = 0
        ## General Parameters
        if d is None:
            d = self.d
        beta_0 = 1 + 0j
        beta_n = self.beta(E, theta)

        wavl = lh.eV2nm(E) * 1e-9
        k0 = 2 * np.pi / wavl
        k0z = k0 * theta
        knz = beta_n * k0z

        ## First Interface matrix
        t_0n = fresnel_t(beta_0, beta_n)
        r_0n = fresnel_r(beta_0, beta_n)
        M_0n = np.array([[1 / t_0n, r_0n / t_0n], [r_0n / t_0n, 1 / t_0n]])  # Transmission and reflection Matrix

        ## Transmission Matrix
        expo_term_pos = np.exp(1j * knz * d)
        expo_term_neg = np.exp(-1j * knz * d)

        P = np.array([[expo_term_pos, 0], [0, expo_term_neg]])  # propagation matrix

        if np.any(~np.isfinite(P)):
            print('Warning! Non-finite values in propagation matrix')
            print(
                f'Beta: {beta_n:.3f}, k0z: {k0z}, Exponent +: {expo_term_pos:.3f}, Exponent : {expo_term_pos:.3f}, P: {P}')

        ## Second interface Matrix
        r_n0 = fresnel_r(beta_n, beta_0)
        t_n0 = fresnel_t(beta_n, beta_0)
        M_n0 = np.array([[1 / t_n0, r_n0 / t_n0], [r_n0 / t_n0, 1 / t_n0]])  # Transmission and reflection Matrix

        L = M_0n @ P @ M_n0
        return L

    def L_final(self, E, theta):
        beta_0 = 1 + 0j
        beta_N = self.beta(E, theta)

        r_N0 = fresnel_r(beta_N, beta_0)
        t_N0 = fresnel_t(beta_N, beta_0)
        M_N0 = np.array([[1 / t_N0, r_N0 / t_N0], [r_N0 / t_N0, 1 / t_N0]])  # Transmission and reflection Matrix

        L = np.array([[0, 0], [0, 1]]) @ M_N0
        return L

    def R(self, E, theta):
        L = self.L(E, theta)
        return -(L[1, 0] / L[1, 1])

    def T(self, E, theta):
        L = self.L(E, theta)
        return L[0, 0] - (L[0, 1] * L[1, 0]) / L[1, 1]