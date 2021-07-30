import xlfluor as xlf
import numpy as np

import scipy.constants as constants
C_r0 = constants.physical_constants['classical electron radius'][0]

DEBUG = True



class Layer:
    """
    Container class for layer data
    """

    def __init__(self, material, thickness, inelastic_cross=0, density=None, final = False):

        if density is not None:  # Specifit density specified, so I need to re-define
            if type(material) is xlf.Element:
                material = xlf.Element(name=material.name, Z=material.Z, atom_weight=material.atom_weight,
                                       density=density)
            elif type(material) is xlf.Composite:
                material = xlf.Composite(name=material.name, elements=material.elements, density=density,
                                         composition=material.composition)
        self.material = material
        self.d = thickness  # nm->m
        self.density = density
        self.sigma_inel = inelastic_cross
        self.is_active = inelastic_cross > 0
        self.min_z = None
        self.max_z = None
        self.z_points = None
        self.dz = None
        self.known_solutions = {}
        self.solution = None
        self.final = final
        self.index = None

        if DEBUG:
            print(f'{self.material.name} Layer Initiated.')

    def solve(self, problem, d: float = None, rho: float = None):
        """
        This function calculates solution matrices for the layer and stores them as a LayerSolution Objext in
        self.solution as well as in the known_solutions dictionary.
        If a solution for this thickness and density is known in the dictionary, it is loaded to self.solution instead
        Also returns the solution.
        :param problem: Problem (mostly necessary for coordinates)
        :param d: thickness
        :param rho: density
        :param final: If true, layer is treated as substate
        :return: LayerSolution
        """

        if rho is not None:
            self.material.update_density(rho)
        if d is not None:
            self.d = d

        solution_key = (self.d, self.material.density)
        if not solution_key in self.known_solutions.keys():
            self.known_solutions[solution_key] = LayerSolution(self, solution_key, problem)

        self.solution = self.known_solutions[solution_key]
        return self.known_solutions[solution_key]

    def beta(self, E, theta):
        wavl_given = xlf.eV2nm(E) * 1e-9  # in meters
        prefactor = self.material.atomar_density * wavl_given ** 2 / (2 * np.pi)
        f = self.material.f(E)
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

        wavl = xlf.eV2nm(E) * 1e-9
        k0 = 2 * np.pi / wavl
        k0z = k0 * theta
        knz = beta_n * k0z

        ## First Interface matrix
        t_0n = xlf.fresnel_t(beta_0, beta_n)
        r_0n = xlf.fresnel_r(beta_0, beta_n)
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
        r_n0 = xlf.fresnel_r(beta_n, beta_0)
        t_n0 = xlf.fresnel_t(beta_n, beta_0)
        M_n0 = np.array([[1 / t_n0, r_n0 / t_n0], [r_n0 / t_n0, 1 / t_n0]])  # Transmission and reflection Matrix

        L = M_0n @ P @ M_n0
        return L

    def L_final(self, E, theta):
        beta_0 = 1 + 0j
        beta_N = self.beta(E, theta)

        r_N0 = xlf.fresnel_r(beta_N, beta_0)
        t_N0 = xlf.fresnel_t(beta_N, beta_0)
        M_N0 = np.array([[1 / t_N0, r_N0 / t_N0], [r_N0 / t_N0, 1 / t_N0]])  # Transmission and reflection Matrix

        L = np.array([[0, 0], [0, 1]]) @ M_N0
        return L

    def R(self, E, theta):
        L = self.L(E, theta)
        return -(L[1, 0] / L[1, 1])

    def T(self, E, theta):
        L = self.L(E, theta)
        return L[0, 0] - (L[0, 1] * L[1, 0]) / L[1, 1]


class LayerSolution:
    """
    Calculates all required single-layer L-matrices in the constructor
    """

    def __init__(self, layer, solution_key, problem):
        self.solution_ID = solution_key
        self.problem = problem

        self.L_matrices_in = np.empty((len(self.problem.energies_in), len(self.problem.angles_in), 2, 2), dtype=complex)
        self.L_matrices_out = np.empty((len(self.problem.energies_out), len(self.problem.angles_out), 2, 2),
                                       dtype=complex)

        # Partial L-matrizes for field within cavity
        self.L_matrices_in_partials = np.empty(
            (len(self.problem.energies_in), len(self.problem.angles_in), len(layer.z_points), 2, 2),
            dtype=complex)
        if layer.is_active:  # Emitted fields are only needed with depth resolution for active layers
            self.L_matrices_out_partials = np.empty(
                (len(self.problem.energies_out), len(self.problem.angles_out), len(layer.z_points), 2, 2),
                dtype=complex)

        ## This is where I can later parallelize
        # Incident field Matrices
        for i_E, E in enumerate(self.problem.energies_in):
            for i_a, angle in enumerate(self.problem.angles_in):
                if layer.final:
                    self.L_matrices_in[i_E, i_a, :, :] = layer.L_final(E, angle)
                else:
                    self.L_matrices_in[i_E, i_a, :, :] = layer.L(E, angle)

                for i_z, z in enumerate(layer.z_points):
                    self.L_matrices_in_partials[i_E, i_a, i_z, :, :] = layer.L(E, angle, z - layer.min_z)

        # Emitted field Matrices
        for i_E, E in enumerate(self.problem.energies_out):
            for i_a, angle in enumerate(self.problem.angles_out):
                if layer.final:
                    self.L_matrices_out[i_E, i_a, :, :] = layer.L_final(E, angle)
                else:
                    self.L_matrices_out[i_E, i_a, :, :] = layer.L(E, angle)

                if layer.is_active:
                    for i_z, z in enumerate(layer.z_points):
                        self.L_matrices_out_partials[i_E, i_a, i_z, :, :] = layer.L(E, angle, z - layer.min_z)

        if DEBUG:
            print('Layer Solution Calculated.')