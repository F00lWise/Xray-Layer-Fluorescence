import numpy as np
import scipy.constants as constants
import os

C_amu = constants.physical_constants['atomic mass constant'][0]
C_r0  = constants.physical_constants['classical electron radius'][0]

class Material:

    def __init__(self, name: str, density: float):
        self.name = name
        self.atom_weight = None
        self.f = None
        self.density = density
        self.atomar_density = self.density / (self.atom_weight * C_amu)


class Element(Material):
    """
    Element data saved in a class.
    method f(E) returns  complex scattering factor (forward) at energy E
    """
    def __init__(self, name: str, Z: int, atom_weight: float, density: float):
        self.Z = Z
        self.name = name
        self.atom_weight = atom_weight
        self.density = density
        global C_amu
        self.atomar_density = self.density / (self.atom_weight * C_amu)

    def _load_scattering_factor(self, element, energy):
        filepath = os.path.join('.','xlfluor','scattering_factor_files',element+'.nff')
        file_as_matrix = np.loadtxt(filepath, skiprows=1)
        energies = file_as_matrix[:, 0]
        f1s = file_as_matrix[:, 1]
        f2s = file_as_matrix[:, 2]
        return np.interp(energy, energies, f1s), np.interp(energy, energies, f2s)

    def f(self, E):
        f1, f2 = self._load_scattering_factor(self.name, E)
        return f1 + 1j * f2


class Composite(Material):
    def __init__(self, name: str, elements, density: float, composition):
        self.name = name
        self.elements = elements
        self.density = density

        assert len(elements) == len(composition), 'Composition must contain one entry per element.'
        elements = np.array(elements)
        self.composition = np.array(composition)
        self._compute_partial_number_densities(self.composition, density)

    def _compute_partial_number_densities(self, composition, density):
        N = len(self.elements)
        self.relative_number_density = self.composition / np.sum(
            self.composition)  # share of each species by relative number of atoms

        average_atomar_weight = np.sum([element.atom_weight * rel_dens \
                                        for element, rel_dens in zip(self.elements, self.relative_number_density)])
        self.atomar_density = density / (average_atomar_weight * C_amu)  # total number of atoms per m3
        self.partial_number_densities = self.relative_number_density * self.atomar_density  # number of atoms of each species per m3

    def f(self, E) -> np.complex:
        all_f = np.array([element.f(E) for element in self.elements], dtype=np.complex)
        return np.sum(self.relative_number_density * all_f)


class Vacuum(Material):
    def __init__(self):
        self.Z = 0
        self.name = 'vacuum'
        self.atom_weight = 0
        self.density = 0
        self.atomar_density = 0

    def f(self, *args):
        return 0 + 0j


