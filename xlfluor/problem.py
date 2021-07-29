import numpy as np
from FLASHutil import little_helpers as lh
from .functions import abs2
import xlfluor

global DEBUG

class Problem:
    """
    This class takes care of the coordinate system that the simulation is performed in.
    """
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

        #### Figure out the z-axis
        z_axis = [0]
        z_layer_indices = [0]  # corresponding to z and contains the index of the layer for each point
        z = 0
        for il, layer in enumerate(self.cavity.layer_list[:-1]):
            layer.min_z = z
            layer.max_z = z+layer.d
            layer_nsteps = passive_layer_resolution if not layer.is_active else active_layer_resolution
            layer.dz = layer.d / layer_nsteps

            z_layer_indices = z_layer_indices + list(np.ones(layer_nsteps) * il)

            layer.z_points = np.linspace(layer.dz/2,layer.d - layer.dz/2, layer_nsteps) # one step left out in the end to avoid a point "between the layers" at interfaces
            z_axis = z_axis + list(z+layer.z_points)
            z = layer.max_z

        z_axis = z_axis + list(np.linspace(z + .5e-9 /2, z + 3e-9, 3))  # Substrate is calculated in 3 depth, 1 nm apart
        z_layer_indices = z_layer_indices + list(np.ones(3) * (len(cavity.layer_list) - 1))

        ### d_thout
        self.d_angle_out = self.angles_out[1] - self.angles_out[0]
        assert np.all(np.diff(self.angles_out) == self.d_angle_out )# Assert that angles out has a constant differential


        #### save axes data
        self.energies_in = self.axes[0]
        self.energies_out = self.axes[1]
        self.angles_in = self.axes[2]
        self.angles_out = self.axes[3]
        self.z_axis = np.array(z_axis)
        self.z_layer_indices = np.array(z_layer_indices, dtype=np.uint8)

        # Some results that should be readily available but ate not directly computed by cavity.solve
        self.reflectivity = np.zeros((len(self.angles_in)), dtype=np.complex)
        self.fluorescence_I_angle_in_dependent = np.zeros((len(problem.angles_in)), dtype=np.complex)
        self.fluorescence_I_angle_out_dependent = np.zeros((len(problem.angles_out)), dtype=np.complex)


        ## Finaly, propose this problem to a cavity:
        cavity.propose_problem((self))

        if DEBUG:
            print('Problem Initiated.')

    def solve(self, cavity: xlfluor.Cavity, parameters):
        assert cavity.solution.problem is self #just check that we are all pointing to the same problem

        cavity.solution.solve(parameters)

        self.reflectivity = cavity.solution.calc_R()
        self.fluorescence_I_angle_in_dependent = xlfluor.functions.abs2( np.sum(cavity.solution.fluorescence_emitted_amplitude, axis=3) )
        self.fluorescence_I_angle_out_dependent = xlfluor.functions.abs2( np.sum(cavity.solution.fluorescence_emitted_amplitude, axis=2) )

        if DEBUG:
            print('Result Initiated.')
