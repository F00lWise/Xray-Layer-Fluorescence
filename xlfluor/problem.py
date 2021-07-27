import numpy as np
from FLASHutil import little_helpers as lh
from .functions import abs2

global DEBUG

class Problem:
    # Nested class to cavity
    def __init__(self, cavity, experiment_data: dict = None, axes: dict = None, passive_layer_resolution: int = 1, active_layer_resolution: int = 10):
        """

        :param cavity:
        :param experiment_data: Either this or axes_data must be given to  build coordinate system
                            example_experiment_data = {
                                'fluor_trace': lh.normmax(wide_scan.fluor_diode),
                                'refl_trace': lh.normmax(wide_scan.refl),
                                'angles_in': np.array(wide_scan['sry']),
                                'angles_out': np.array(wide_scan['sry']),
                                'energies_in': np.array([wide_scan['energy'], ]),
                                'energies_out': np.array([6400, ])
                            }
        :param axes: Either this or axes_data must be given to  build coordinate system
        """
        ## store cavity
        self.cavity = cavity

        #### Generate coordinates to calculate everything in
        if experiment_data is not None:
            self.axes = experiment_data['energies_in'], experiment_data['energies_out'], experiment_data['angles_in'], experiment_data['angles_out']
            if axes is not None:
                assert axes == self.axes
        elif axes is not None:
            self.axes = axes
        else:
            raise ValueError('To generate axes I either need experimental data or specific axes')

        #### Figure out the axis in depth
        z_axis = [0]
        z_layer_indices = [0]  # corresponding to z and contains the index of the layer for each point
        z = 0
        for il, layer in enumerate(self.cavity.layer_list[:-1]):
            layer.min_z = z
            layer.max_z = z+layer.d

            z_resolution = layer.d / layer_nsteps

            layer_nsteps = passive_layer_resolution if not layer.is_active else active_layer_resolution
            z_layer_indices = z_layer_indices + list(np.ones(layer_nsteps) * il)

            layer.z_points = np.linspace(z_resolution/2,layer.d - z_resolution/2, layer_nsteps) # one step left out in the end to avoid a point "between the layers" at interfaces
            z_axis = z_axis + list(z+layer.z_points)
            z = layer.max_z

        z_axis = z_axis + list(np.linspace(z + .5e-9 /2, z + 3e-9, 3))  # Substrate is calculated in 3 depth, 1 nm apart
        z_edges = lh.edgepoints(np.array(z_axis))

        z_layer_indices = z_layer_indices + list(np.ones(3) * (len(cavity.layer_list) - 1))

        #### save axes data
        self.energies_in = self.axes[0]
        self.energies_out = self.axes[1]
        self.angles_in = self.axes[2]
        self.angles_out = self.axes[3]
        self.z_edges = np.array(z_edges)
        self.z_axis = np.array(z_axis)
        self.z_layer_indizes = np.array(z_layer_indices, dtype=np.uint8)

        self.solution = ProblemSolution(self)

        if DEBUG:
            print('Problem Initiated.')


    def update(self, parameters):
        ### Update layer parameters

        self.solution.update(self)

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



class ProblemSolution:
    def __init__(self, problem):
        ## Initiate L matrizes
        self.L_matrices_in = np.zeros((4,4,len(problem.energies_in),len(problem.angles_in)), dtype = complex)
        self.L_matrices_out = np.zeros((4,4,len(problem.energies_out),len(problem.angles_out)), dtype = complex)

        #Partial L-matrizes for field within cavity
        self.L_matrices_in_partials = np.zeros((4,4,len(problem.energies_in),len(problem.angles_in),problem.z_axis), dtype = complex)
        self.L_matrices_out_partials = np.zeros((4,4,len(problem.energies_out),len(problem.angles_out),problem.z_axis), dtype = complex)

        self.layer_solutions = [layer.solve(self) for layer in problem.cavity.layer_list[:-1]] # calculate initial layer solutions
        self.layer_solutions.append(problem.cavity.layer_list[-1].solve())

        self.result = Result(problem)

        if DEBUG:
            print('ProblemSolution Initiated.')
    def update(self, problem):
        for i_layer, layer in enumerate(problem.cavity.layer_list):
            self.layer_solutions[i_layer] = layer.solve(self, d=, rho=)



class Result:
    def __init__(self, problem):
        #### Initialize Result
        self.incident_fields = np.zeros((len(problem.z_axis), len(problem.angles_in)), dtype=np.complex)
        self.reflectivity = np.zeros((len(problem.angles_in)), dtype=np.complex)
        self.fluor_emitted = np.zeros((len(problem.angles_in), len(problem.angles_out)), dtype=np.complex)
        self.fluor_emitted_total = np.zeros((len(problem.angles_in)), dtype=np.complex)
        self.fluor_emitted_from_z = np.zeros((len(problem.z_axis), len(problem.angles_in), len(problem.angles_out)),
                                             dtype=np.complex)
        self.fluor_local_amplitude = np.zeros((len(problem.z_axis), len(problem.angles_in), len(problem.angles_out), 2),
                                              dtype=np.complex)

        if DEBUG:
            print('Result Initiated.')