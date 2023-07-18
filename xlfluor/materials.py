import numpy as np
import scipy.constants as constants
import os

DEBUG = False

C_amu = constants.physical_constants['atomic mass constant'][0]
C_r0  = constants.physical_constants['classical electron radius'][0]

class Material:
    """
    Abstract Material class.

    Attributes:
        name (str): Name of the material.
        density (float): Density of the material.

    Methods:
        __init__(self, name: str, density: float):
            Initialize the Material object.
    """
    def __init__(self, name: str, density: float):
        self.name = name
        self.density = density
        if DEBUG:
            print(f'Material {self.name} Initiating.')

class Element(Material):
    """
    Element data saved in a class. Method f(E) returns complex scattering factor (forward) at energy E.

    Attributes:
        Z (int): Atomic number of the element.
        atom_weight (float): Atomic weight of the element.
        atomar_density (float): Atomar density of the element.
        scattering_factor_file: File object storing the scattering factor data.

    Methods:
        __init__(self, name: str, Z: int, atom_weight: float, density: float):
            Initialize the Element object.
        _load_scattering_factor(self, element, energy):
            Load the scattering factor data for the element at the given energy.
        update_density(self, new_density):
            Update the density of the element.
        f(self, E):
            Calculate the complex scattering factor (forward) of the element at the given energy.
    """

    def __init__(self, name: str, Z: int, atom_weight: float, density: float):
        super().__init__(name, density)
        self.Z = Z
        self.atom_weight = atom_weight
        self.atomar_density = self.density / (self.atom_weight * C_amu)

        self.scattering_factor_file = None

    def _load_scattering_factor(self, element, energy):
        """
        Load the scattering factor data for the element at the given energy.

        :param element: Element name.
        :param energy: Photon energy.
        :return: Scattering factor values (f1, f2).
        """
        if self.scattering_factor_file is None:
            filepath = os.path.join('.','xlfluor','scattering_factor_files',element+'.nff')
            self.scattering_factor_file = np.loadtxt(filepath, skiprows=1)
        energies = self.scattering_factor_file[:, 0]
        f1s = self.scattering_factor_file[:, 1]
        f2s = self.scattering_factor_file[:, 2]
        return np.interp(energy, energies, f1s), np.interp(energy, energies, f2s)

    def update_density(self, new_density):
        """
        Update the density of the element, calculating the atomar density again.
        This is useful during fitting routines so as to avoid setting up the class anew.

        :param new_density: New density value.
        """
        self.density = new_density
        self.atomar_density = new_density / (self.atom_weight * C_amu)


    def f(self, E):
        """
        Calculate the complex scattering factor (forward) of the element at the given energy.

        :param E: Photon energy.
        :return: Complex scattering factor (f).
        """
        f1, f2 = self._load_scattering_factor(self.name, E)
        return f1 + 1j * f2


class Composite(Material):
    """
    Composite material class for materials composed of several elements.

    Attributes:
        elements (list): List of Element objects.
        composition (list): List of composition values corresponding to each element.
        relative_number_density: Relative number density of each element.
        average_atomar_weight: Average atomar weight of the composite material.
        partial_number_densities: Number density of each element.

    Methods:
        __init__(self, name: str, elements, density: float, composition: list):
            Initialize the Composite object.
        _compute_partial_number_densities(self, density):
            Compute the partial number densities of the composite material.
        update_density(self, new_density):
            Update the density of the composite material.
        f(self, E) -> np.complex128:
            Calculate the complex scattering factor (forward) of the composite material at the given energy.
    """
    def __init__(self, name: str, elements, density: float, composition: list):
        super().__init__(name, density)
        self.elements = elements

        assert len(elements) == len(composition), 'Composition must contain one entry per element.'
        elements = np.array(elements)
        self.composition = np.array(composition)
        self._compute_partial_number_densities(density)

    def _compute_partial_number_densities(self, density):
        """
        Compute the partial number densities of the composite material.

        :param density: Density of the composite material.
        """
        N = len(self.elements)
        self.relative_number_density = self.composition / np.sum(self.composition)  # share of each species by relative number of atoms

        self.average_atomar_weight = np.sum([element.atom_weight * rel_dens \
                                        for element, rel_dens in zip(self.elements, self.relative_number_density)])

        self.atomar_density = density / (self.average_atomar_weight * C_amu)  # total number of atoms per m3
        self.partial_number_densities = self.relative_number_density * self.atomar_density  # number of atoms of each species per m3

    def update_density(self, new_density):
        """
        Update the density of the composite material.

        :param new_density: New density value.
        """
        self.density = new_density
        self.atomar_density = new_density / (self.average_atomar_weight * C_amu)  # total number of atoms per m3
        self.partial_number_densities = self.relative_number_density * self.atomar_density  # number of atoms of each species per m3

    def f(self, E) -> np.complex128:
        """
        Calculate the complex scattering factor (forward) of the composite material at the given energy.

        :param E: Photon energy.
        :return: Complex scattering factor (f).
        """
        all_f = np.array([element.f(E) for element in self.elements], dtype=np.complex128)
        return np.sum(self.relative_number_density * all_f)


class Vacuum(Material):
    """
    Vacuum material class representing empty space.

    Attributes:
        Z (int): Atomic number of the vacuum.
        atom_weight (float): Atomic weight of the vacuum.
        atomar_density (float): Atomar density of the vacuum.

    Methods:
        __init__(self, name: str = 'vacuum'):
            Initialize the Vacuum object.
        f(self, *args):
            Calculate the complex scattering factor (forward) of the vacuum.
    """
    def __init__(self, name: str= 'vacuum'):
        super().__init__(name=name, density=0)
        self.Z = 0
        self.atom_weight = 0
        self.atomar_density = 0

    def f(self, *args):
        """
        Return the complex scattering factor of the vacuum (it is 0 ;-) ).

        :param args: Placeholder for additional arguments.
        :return: Complex scattering factor (f) of the vacuum.
        """
        return 0 + 0j


