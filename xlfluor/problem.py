import numpy as np
import xlfluor as xlf
import concurrent.futures
DEBUG = False


class Problem:
    """
    This class takes care of the coordinate system that the simulation is performed in.
    """
    def __init__(self, cavity, experiment_data: dict = None, axes: tuple = None, passive_layer_resolution: int = 1, active_layer_resolution: int = 10):
        """

        :param cavity:
        :param experiment_data: Either this or axes_data must be given to  build coordinate system
            Axes are taken from experiment data if given there, otherwise taken from axes.
                experiment_data = {
                    'fluor_trace': xlf.normmax(loaded_scan['fluor_diode']),
                    'refl_trace': xlf.normmax(loaded_scan['refl']),
                    'angles_in': np.array(loaded_scan['sry']),
                    'energies_in': np.array([loaded_scan['energy'], ]),
                    'energies_out': np.array([6400, ])
                }
        :param axes: Either this or axes_data must be given to  build coordinate system
        """
        ### Initialize multiprocessing pool
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=cavity.N_layers)

        ## store cavity
        self.cavity = cavity

        ## store experiment
        self.experiment = experiment_data
        #### Generate coordinates to calculate everything in
        axes_list= []
        for i, axname in enumerate( ['energies_in','energies_out','angles_in','angles_out']):
            if experiment_data is not None and axname in experiment_data.keys():
                axes_list.append(experiment_data[axname])
            elif axes is not None and axname in axes.keys():
                axes_list.append(axes[axname])
            else:
                raise ValueError(f'Axes {axname} found neither in the experiment data nor in the axes dictionary')
        self.axes = tuple(axes_list)

        #### Figure out the z-axis
        z_axis = []
        z_layer_indices = []  # corresponding to z and contains the index of the layer for each point
        z = 0

        final_layer_calcuated_depth = 3e-9

        for il, layer in enumerate(self.cavity.layer_list[:]):
            layer.min_z = z
            if not layer.final:
                layer_nsteps = passive_layer_resolution if not layer.is_active else active_layer_resolution
                layer.max_z = z+layer.d
                layer.dz = layer.d / layer_nsteps
                layer.z_points = np.linspace(layer.dz / 2, layer.d - layer.dz / 2,
                                             layer_nsteps)  # one step left out in the end to avoid a point "between the layers" at interfaces
            else: #final layer
                layer_nsteps = 3
                layer.max_z = np.inf
                layer.dz = 1e-9
                layer.z_points = np.linspace(layer.dz / 2, final_layer_calcuated_depth - layer.dz / 2,
                                             layer_nsteps)  # one step left out in the end to avoid a point "between the layers" at interfaces

            z_layer_indices = z_layer_indices + list(np.ones(layer_nsteps) * il)

            z_axis = z_axis + list(z+layer.z_points)
            z = layer.max_z


        #### save axes data
        self.energies_in = self.axes[0]
        self.energies_out = self.axes[1]
        self.angles_in = self.axes[2]
        self.angles_out = self.axes[3]
        self.z_axis = np.array(z_axis)
        self.z_layer_indices = np.array(z_layer_indices, dtype=np.uint8)
        ### d_thout
        self.d_angle_out = self.angles_out[1] - self.angles_out[0]
        assert np.all(np.diff(self.angles_out) - self.d_angle_out <1e-13)# Assert that angles out has a constant differential

        # Some results that should be readily available but ate not directly computed by cavity.solve
        self.reflectivity = np.zeros((len(self.angles_in)), dtype=np.complex)
        self.fluorescence_I_angle_in_dependent = np.zeros((len(self.angles_in)), dtype=np.complex)
        self.fluorescence_I_angle_out_dependent = np.zeros((len(self.angles_out)), dtype=np.complex)


        ## At first calculate the full fields
        self.full_field_solution = True
        
        ## Finaly, propose this problem to a cavity:
        cavity.propose_problem(self)
        
        ## Normally only calculate field strength within the cavity that are relevant for the emitted flurescence
        self.full_field_solution = False
        
        if DEBUG:
            print('Problem Initiated.')


    def __del__(self):
        print("Destructing the problem.")
        self.executor.shutdown()

    def __repr__(self):
        return f'Problem({id(self)})'

    def solve(self, cavity, parameters,calculate_full_fields= False):
        assert cavity.solution.problem is self #just check that we are all pointing to the same problem
        self.full_field_solution = calculate_full_fields
        
        cavity.solution.solve(parameters)

        self.reflectivity = cavity.solution.calc_R()
        self.fluorescence_I_angle_in_dependent = np.sum(cavity.solution.fluorescence_emitted_intensity, axis=(0,1,3))
        self.fluorescence_I_angle_out_dependent = np.sum(cavity.solution.fluorescence_emitted_intensity, axis=(0,1,2))
        if DEBUG:
            print('Result Initiated.')
