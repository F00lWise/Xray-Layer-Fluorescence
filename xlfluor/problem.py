import numpy as np
from FLASHutil import little_helpers as lh
from .functions import abs2

class Problem:
    # Nested class to cavity
    def __init__(self, cavity, energy_in, energy_out, angles_in, angles_out):
        ## store cavity
        self.cavity = cavity

        #### Figure out the axis in depth
        z_edges = []
        z_layer_indizes = []  # corresponding to z and contains the index of the layer for each point
        z = 0
        for il, layer in enumerate(cavity.layer_list[:-1]):
            z_layer_indizes = z_layer_indizes + list(np.ones(layer.d_steps) * il)
            z_edges = z_edges + list(np.linspace(z, z + layer.d - (layer.d / layer.d_steps),
                                                 layer.d_steps))  # one step left out in the end to avoid a point "between the layers" at interfaces
            z = z + layer.d

        z_edges = z_edges + list(np.linspace(z, z + 3e-9, 3))  # Substrate is calculated in 3 depth, 1 nm apart
        z_axis = lh.midpoints(np.array(z_edges))
        z_layer_indizes = z_layer_indizes + list(np.ones(3) * (len(cavity.layer_list) - 1))

        #### save axes data
        self.energy_in = energy_in
        self.energy_out = energy_out
        self.angles_in = angles_in
        self.angles_out = angles_out
        self.z_edges = np.array(z_edges)
        self.z_axis = np.array(z_axis)
        self.z_layer_indizes = np.array(z_layer_indizes, dtype=np.uint8)

        #### Initialize Solution
        #
        self.incident_fields = np.zeros((len(z_axis), len(angles_in)), dtype=np.complex)
        self.reflectivity = np.zeros((len(angles_in)), dtype=np.complex)

        self.fluor_emitted = np.zeros((len(angles_in), len(angles_out)), dtype=np.complex)
        self.fluor_emitted_from_z = np.zeros((len(z_axis), len(angles_in), len(angles_out)), dtype=np.complex)
        self.fluor_local_amplitude = np.zeros((len(z_axis), len(angles_in), len(angles_out), 2), dtype=np.complex)

    def solve(self):
        for iai, angle_in in enumerate(self.angles_in):
            self.reflectivity[iai] = self.cavity.R(self.energy_in, angle_in)

            self.fluor_emitted[iai, :] = 0
            for iz, z in enumerate(self.z_axis * 1e9):
                layer_index = self.z_layer_indizes[iz]  # layer are we in at this z

                self.incident_fields[iz, iai] = self.cavity.field(self.energy_in, angle_in, z)

                ### calculate_fluoreszens from this z
                if self.cavity.layer_list[layer_index].sigma_inel > 0:

                    # self.incident_fields[iz,iai] = self.cavity.field(self.energy_in,angle_in,z)

                    for iao, angle_out in enumerate(self.angles_out):
                        self.fluor_emitted_from_z[iz, iai, iao], self.fluor_local_amplitude[iz, iai, iao, :] = \
                            self.cavity.fluor(self.energy_out, \
                                              angle_out, \
                                              z, \
                                              self.incident_fields[iz, iai], \
                                              self.cavity.layer_list[layer_index].sigma_inel)

                        # print(f'Fluorescence from Eout = {self.energy_out}, angle = {angle_out}, z ={z}, field = {self.incident_fields[iz,iai]}, sigma = {Cavity.layer_list[layer_index].sigma_inel}')
                        # print(f'{ self.fluor_emitted_from_z[iz,iai,iao,:]} {self.fluor_local_amplitude[iz,iai,:,:]}')
                        # self.fluor_emitted[iai,:]  = self.fluor_emitted[iai,:]+\
                        #    self.fluor_emitted_from_z[iz,iai,:] * (z-z_axis[zi-1])

            z_differences = np.diff(self.z_edges)
            self.fluor_emitted[iai, :] = np.sum(self.fluor_emitted_from_z[:, iai, :] * \
                                                np.outer(z_differences, np.ones(len(self.angles_out))), 0)

        # Calculate projections
        self.fluotescence_I_angle_out_dependent = np.sum(abs2(self.fluor_emitted), 0)
        self.fluotescence_I_angle_in_dependent = np.sum(abs2(self.fluor_emitted), 1)

        # Since I multiplied each contribution in z with the dz to weigh the contributions by thickness i now divide by the total
        # (Double-check units?)
        total_active_thickness = np.sum([layer.d for layer in self.cavity.layer_list if layer.sigma_inel > 0])
        self.fluor_emitted = self.fluor_emitted / total_active_thickness