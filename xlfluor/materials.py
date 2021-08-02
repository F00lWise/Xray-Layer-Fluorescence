import numpy as np
import scipy.constants as constants
import os

DEBUG = False

C_amu = constants.physical_constants['atomic mass constant'][0]
C_r0  = constants.physical_constants['classical electron radius'][0]

class Material:
    """
    Abstract Material class
    """
    def __init__(self, name: str, density: float):
        self.name = name
        self.density = density
        if DEBUG:
            print(f'Material {self.name} Initiating.')

class Element(Material):
    """
    Element data saved in a class.
    method f(E) returns  complex scattering factor (forward) at energy E
    """
    def __init__(self, name: str, Z: int, atom_weight: float, density: float):
        super().__init__(name, density)
        self.Z = Z
        self.atom_weight = atom_weight
        self.atomar_density = self.density / (self.atom_weight * C_amu)

        self.scattering_factor_file = None

    def _load_scattering_factor(self, element, energy):
        if self.scattering_factor_file is None:
            filepath = os.path.join('.','xlfluor','scattering_factor_files',element+'.nff')
            self.scattering_factor_file = np.loadtxt(filepath, skiprows=1)
        energies = self.scattering_factor_file[:, 0]
        f1s = self.scattering_factor_file[:, 1]
        f2s = self.scattering_factor_file[:, 2]
        return np.interp(energy, energies, f1s), np.interp(energy, energies, f2s)

    def update_density(self, new_density):
        self.density = new_density
        self.atomar_density = new_density / (self.atom_weight * C_amu)


    def f(self, E):
        f1, f2 = self._load_scattering_factor(self.name, E)
        return f1 + 1j * f2


class Composite(Material):
    def __init__(self, name: str, elements, density: float, composition: list):
        super().__init__(name, density)
        self.elements = elements

        assert len(elements) == len(composition), 'Composition must contain one entry per element.'
        elements = np.array(elements)
        self.composition = np.array(composition)
        self._compute_partial_number_densities(density)

    def _compute_partial_number_densities(self, density):
        N = len(self.elements)
        self.relative_number_density = self.composition / np.sum(self.composition)  # share of each species by relative number of atoms

        self.average_atomar_weight = np.sum([element.atom_weight * rel_dens \
                                        for element, rel_dens in zip(self.elements, self.relative_number_density)])

        self.atomar_density = density / (self.average_atomar_weight * C_amu)  # total number of atoms per m3
        self.partial_number_densities = self.relative_number_density * self.atomar_density  # number of atoms of each species per m3

    def update_density(self, new_density):
        self.density = new_density
        self.atomar_density = new_density / (self.average_atomar_weight * C_amu)  # total number of atoms per m3
        self.partial_number_densities = self.relative_number_density * self.atomar_density  # number of atoms of each species per m3

    def f(self, E) -> np.complex:
        all_f = np.array([element.f(E) for element in self.elements], dtype=np.complex)
        return np.sum(self.relative_number_density * all_f)


class Vacuum(Material):
    def __init__(self, name: str= 'vacuum'):
        super().__init__(name=name, density=0)
        self.Z = 0
        self.atom_weight = 0
        self.atomar_density = 0

    def f(self, *args):
        return 0 + 0j


