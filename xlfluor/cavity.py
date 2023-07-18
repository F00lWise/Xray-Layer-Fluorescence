import numpy as np
import lmfit
import xlfluor as xlf

import concurrent.futures
import time
DEBUG = True

def timeit(func):
    """
    Decorator for measuring function's running time.
    """
    def measure_time(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        print("Processing time of %s(): %.3f milliseconds."
              % (func.__qualname__, 1e3*(time.time() - start_time)))
        return result

    return measure_time

class Cavity:
    """
    Represents an optical cavity system.

    Attributes:
        N_layers (int): Number of layers in the cavity.
        layer_list (list): List of Layer objects representing the layers in the cavity.
        solution (CavitySolution): Solution object associated with the cavity.

    Methods:
        propose_problem(problem: Problem) -> None:
            Proposes a problem to the cavity, linking the cavity's solution to the problem.

        solve(problem: Problem) -> None:
            Solves the problem associated with the cavity.

        calc_R() -> np.ndarray:
            Calculates the reflectivity for each input angle based on the solved problem.

        calc_T() -> np.ndarray:
            Calculates the transmittance for each input angle based on the solved problem.
    """
    def __init__(self, layer_list: list):
        self.D = np.sum([layer.d for layer in layer_list[:-1]])
        self.layer_list = layer_list
        self.N_layers = len(layer_list)
        self.parameters = self.layer_list_to_parameters(self.layer_list)

        self.solution = None

        for n, layer in enumerate(layer_list):
            layer.index = n

        global DEBUG
        if DEBUG:
            print('Cavity Initiated.')

    def propose_problem(self, problem) -> None:
        """
        Takes a problem and offers a solution
        :param problem:
        :return:
        """
        self.solution = CavitySolution(self, problem)

    def layer_list_to_parameters(self,layer_list, d_tolerance=2e-9, rho_tolerance=0.5e3):
        """
        takes in a parameter list and generates a layer list from it which can be treated by an optimizer
        :param d_tolerance:
        :param layer_list:
        :return:
        """
        params = lmfit.Parameters()
        for n, layer in enumerate(layer_list):
            prefix = f'Layer_{n}_{layer.material.name}_'
            if not layer.final:
                params.add(prefix + 'd', value=layer.d, min=np.max([layer.d - d_tolerance,0]), max=layer.d + d_tolerance)
            else:
                params.add(prefix + 'd', value=layer.d, vary=False)

            params.add(prefix + 'rho', value=layer.density, min=np.max([layer.density - rho_tolerance,0]),
                     max=layer.density + rho_tolerance)
        return params
    
    def get_relative_intensities(self, problem):
   
        I_fluor = np.max(problem.fluorescence_I_angle_in_dependent)
        I_refl = np.max(xlf.abs2(problem.reflectivity)[0, :])

        # Scaling parameters - multiplier to the  uncalibrated and normalized experimental data to fit to the model data
        self.parameters.add('I_fluorescence', value = I_fluor, min = I_fluor/10, max =I_fluor*10)
        self.parameters.add('I_reflectivity', value = I_refl, min = I_refl/10, max =I_refl*10)

        print(f'Based on the Initial fit, setting relative intensity of refl and fluor \n to {I_refl} and {I_fluor}, respectively')

    def set_fit_weigths(self, weight_refl = 1, weight_fluor = 1):
        self.parameters.add('weight_reflectivity', value = weight_refl, vary = False)
        self.parameters.add('weight_fluorescence', value = weight_fluor, vary = False)

class CavitySolution:
    """
    This is where all the calculations over several layers happen
    """
    def __init__(self, cavity, problem):
        self.problem = problem
        self.cavity = cavity

        ## Initiate L matrizes
        self.L_matrices_in = np.zeros((len(problem.energies_in), len(problem.angles_in), 2, 2), dtype=complex)
        self.L_matrices_out = np.zeros((len(problem.energies_out), len(problem.angles_out), 2, 2), dtype=complex)

        # Partial L-matrizes for field within cavity
        self.L_matrices_in_partials = np.zeros((len(problem.energies_in), len(problem.angles_in), len(problem.z_axis), 2, 2),
                                               dtype=complex)
        self.L_matrices_out_partials = np.zeros(
            (len(problem.energies_out), len(problem.angles_out), len(problem.z_axis), 2, 2), dtype=complex) # L1 or L(z_p)
        self.L_matrices_out_partials_inverse = np.zeros(
            (len(problem.energies_out), len(problem.angles_out), len(problem.z_axis),2,2), dtype=complex)  #L1^-1
        self.L_matrices_out_partials_reverse = np.empty(
            (len(problem.energies_out), len(problem.angles_out), len(problem.z_axis),2,2), dtype=complex)*np.nan  #Lz or L(D-z_p)

        # Field strengths
        self.incident_field_amplitude                = np.empty((len(problem.energies_in), len(problem.angles_in), len(problem.z_axis)), dtype=complex)*np.nan # scalar at each point in input angle, input energy and z
        self.emitted_field_amplitude                = np.empty((len(problem.energies_out), len(problem.angles_out), len(problem.z_axis)), dtype=complex)*np.nan # scalar at each point in input angle, input energy and z
        self.fluorescence_local_amplitude            = np.empty((len(problem.energies_in), len(problem.energies_out),len(problem.angles_in), len(problem.angles_out),len(problem.z_axis), 2), dtype=complex)*np.nan # vector (down, up) at each point
        self.fluorescence_local_amplitude_propagated = np.empty((len(problem.energies_in), len(problem.energies_out),len(problem.angles_in), len(problem.angles_out),len(problem.z_axis)), dtype=complex)*np.nan # fluorescence amplitude from each point in z but propagated to surface
        self.fluorescence_emitted_intensity          = np.empty((len(problem.energies_in), len(problem.energies_out),len(problem.angles_in), len(problem.angles_out)), dtype=complex)*np.nan# scalar (emitted upwards from z=0) at each point in input angle and input energy

        self.layer_solutions = [layer.solve(self.problem) for layer in
                                cavity.layer_list[:-1]]  # calculate initial layer solutions
        self.layer_solutions.append(cavity.layer_list[-1].solve(self.problem))

        self.incident_field_calculated = False
        self.emitted_field_calculated = False

        if DEBUG:
            print('ProblemSolution Initiated.')

    def solve(self, parameters):
        """
        This class does the heavy lifting. One call to this calculates the entire cavity with new parameters
        :param parameters:
        :return:
        """
        self._solve_layers(parameters)

        self._calc_L_total()
        self._calc_L_partial()

        if self.problem.full_field_solution:
            # Otherwise this is done only for the necessary depths z within calc_fluorescence
            self.calc_incident_field()
            self.calc_emitted_field()

        self._calc_fluorescence()

    #@timeit
    def _solve_layers(self, parameters):
        """
        Call all layer.solve() functions with new parameters.
        In the process of layer.solve, the new parameters are also written into the layer parameters
        :param parameters:
        :return:
        """
        full = self.problem.full_field_solution
# Point to parallelize
        futures_to_layer_results = {}
        for n, layer in enumerate(self.problem.cavity.layer_list):
            prefix = f'Layer_{n}_{layer.material.name}_'
            d = parameters[prefix + 'd'].value
            rho = parameters[prefix + 'rho'].value
            #self.layer_solutions[n] = layer.solve(self.problem, d, rho) # This is the code without parallelization
            futures_to_layer_results[self.problem.executor.submit(layer.solve,self.problem, d, rho)] = n

        #print('futures submitted')
        for future in concurrent.futures.as_completed(futures_to_layer_results):
            n = futures_to_layer_results[future]
            try:
                self.layer_solutions[n] = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (n, exc))
             
        # reset possibly calculated results
        self.incident_field_calculated = False
        self.emitted_field_calculated = False


    #@timeit
    def _calc_L_total(self):
        """
        Calculates the transfer matrix of the entire cavity
        :return:
        """
        self.L_matrices_in[...] = np.eye(2)
        self.L_matrices_out[...] = np.eye(2)

        #Iterates through layers and multiplies the layer matrices to get the total cavity matrix
        for n, layer in enumerate(self.cavity.layer_list):
            self.L_matrices_in = layer.solution.L_matrices_in @ self.L_matrices_in
            self.L_matrices_out = layer.solution.L_matrices_out @ self.L_matrices_out

    #@timeit
    def _calc_L_partial(self):
        """
        Calculates the transfer matrix of the cavity up to the specific depths in problem.z_axis
        :return:
        """
        self.L_matrices_in_partials[...] = np.eye(2)
        self.L_matrices_out_partials[...] = np.eye(2)

        #Iterates through layers and multiplies the layer matrices up to the active layer, then breaks that one up and multiplies it as well
        for iz, z in enumerate(self.problem.z_axis):
            # which layer are we in?
            partial_layer_index: int = self.problem.z_layer_indices[iz]
            partial_layer = self.cavity.layer_list[partial_layer_index]

            # Cavity Matrix up to layer that will be split up
            for n, layer in enumerate(self.cavity.layer_list[:partial_layer_index]):
                self.L_matrices_in_partials[:,:,iz,:,:] = layer.solution.L_matrices_in[:,:,:,:]  @ self.L_matrices_in_partials[:,:,iz,:,:]
                if partial_layer.is_active or self.problem.full_field_solution:
                    self.L_matrices_out_partials[:,:,iz,:,:] = layer.solution.L_matrices_out[:,:,:,:] @ self.L_matrices_out_partials[:,:,iz,:,:]

            # Matrix of partial layer
            iz_within_layer = iz - np.where(self.problem.z_layer_indices == partial_layer_index)[0][0] # where statement finds the first z-index of the current partial layer
            self.L_matrices_in_partials[:,:,iz,:,:] = partial_layer.solution.L_matrices_in_partials[:,:,iz_within_layer,:,:]  @ self.L_matrices_in_partials[:,:,iz,:,:]
            if partial_layer.is_active or self.problem.full_field_solution:
                self.L_matrices_out_partials[:,:,iz,:,:] = partial_layer.solution.L_matrices_out_partials[:,:,iz_within_layer,:,:] @ self.L_matrices_out_partials[:,:,iz,:,:]

            # calculate reverse matrices: This equates to: L_2 = L_D @ L_1^-1
            if partial_layer.is_active or self.problem.full_field_solution:
                self.L_matrices_out_partials_inverse[:,:,iz,:,:] = np.linalg.inv(self.L_matrices_out_partials[:,:,iz,:,:]) #L(z_p)^-1
                self.L_matrices_out_partials_reverse[:,:,iz,:,:] = self.L_matrices_out @ self.L_matrices_out_partials_inverse[:,:,iz,:,:] # L2 = L(D) @ L ^-1


    def consistency_check(self):
        Assembled_L = self.L_matrices_out_partials_reverse @ self.L_matrices_out_partials

        imprecision = np.empty(self.L_matrices_out_partials_reverse.shape, dtype = complex)
        for iz,z in enumerate(self.problem.z_axis):
            imprecision[:,:,iz,:,:] = Assembled_L[:,:,iz,:,:] - self.L_matrices_out
        return imprecision

    def calc_R(self) -> np.ndarray:
        """
        Calculates the reflectivity for each input angle based on the last solve() call

        Returns:
            ndarray: Array representing the calculated reflectivity for each input angle.
        """

        return -(self.L_matrices_in[:,:,1, 0] / self.L_matrices_in[:,:,1, 1])

    def calc_T(self) -> np.ndarray:
        """
        Calculates the transmittance for each input angle based on the last solve() call

        Returns:
            ndarray: Array representing the calculated transmittance for each input angle.
        """
        return self.L_matrices_in[0, 0] - (self.L_matrices_in[0, 1] * self.L_matrices_in[1, 0]) / self.L_matrices_in[1, 1]

    #@timeit
    def calc_incident_field(self):
        """
        Calculate the field strength of the incident wave at all depth in z_axis
        """
        
        self.incident_field_amplitude[:,:,:]= self.L_matrices_in_partials[:, :, :, 0, 0] + self.L_matrices_in_partials[:, :, :, 1, 0] - \
                                                         ((self.L_matrices_in_partials[:,:,:,0,1] + self.L_matrices_in_partials[:,:,:,1,1]).transpose(2,0,1) *
                                                          self.L_matrices_in[:,:,1,0] / self.L_matrices_in[:,:,1,1]).transpose(1,2,0)

        self.incident_field_calculated = True
        
    def calc_emitted_field(self):
        """
        Calculate the field strength of the emitted wave at all depth in z_axis
        This is for the full cavity and only needed 
        """
        self.emitted_field_amplitude[:,:,:]  = self.L_matrices_out_partials[:, :, :, 0, 0] + self.L_matrices_out_partials[:, :, :, 1, 0] - \
                                ((self.L_matrices_out_partials[:,:,:,0,1] + self.L_matrices_out_partials[:,:,:,1,1]).transpose(2,0,1) *
                                 self.L_matrices_out[:,:,1,0] / self.L_matrices_out[:,:,1,1]).transpose(1,2,0)
        self.emitted_field_calculated = True
            
    def _calc_fluorescence(self):

        # We are only interested in active layers
        for n, layer in enumerate(self.cavity.layer_list):
            if not layer.is_active:
                continue
            relevant_z_indices = np.where(self.problem.z_layer_indices == n)[0]

            #excitation_intensity = xlf.abs2(self.incident_field_amplitude[:,:,relevant_z_indices])       # Ein,Ain,z(layer)
            #fluorescence_intensity = excitation_intensity * layer.sigma_inel * layer.dz # Ein,Ain,z(layer)
            if not self.incident_field_calculated:               
                self.incident_field_amplitude[:,:,relevant_z_indices]  = self.L_matrices_in_partials[:, :, relevant_z_indices, 0, 0] + self.L_matrices_in_partials[:, :, relevant_z_indices, 1, 0] - \
                                                         ((self.L_matrices_in_partials[:,:,relevant_z_indices,0,1] + self.L_matrices_in_partials[:,:,relevant_z_indices,1,1]).transpose(2,0,1) *
                                                          self.L_matrices_in[:,:,1,0] / self.L_matrices_in[:,:,1,1]).transpose(1,2,0)
            
            fluorescence_amplitude = np.sqrt(xlf.abs2(self.incident_field_amplitude[:,:,relevant_z_indices])  * layer.sigma_inel * layer.dz) # np.sqrt(fluorescence_intensity)   
         
            if not self.emitted_field_calculated:
                self.emitted_field_amplitude[:,:,relevant_z_indices]  = self.L_matrices_out_partials[:, :, relevant_z_indices, 0, 0] + self.L_matrices_out_partials[:, :, relevant_z_indices, 1, 0] - \
                                                         ((self.L_matrices_out_partials[:,:,relevant_z_indices,0,1] + self.L_matrices_out_partials[:,:,relevant_z_indices,1,1]).transpose(2,0,1) *
                                                          self.L_matrices_out[:,:,1,0] / self.L_matrices_out[:,:,1,1]).transpose(1,2,0)
            
          
            # Non-parallel code
            for iEin, Ein in enumerate(self.problem.energies_in):
                for iAin, Ain in enumerate(self.problem.angles_in):
                    self.fluorescence_local_amplitude_propagated[iEin, :,iAin ,:, relevant_z_indices] = (self.emitted_field_amplitude[:,:,relevant_z_indices]  * fluorescence_amplitude[iEin, iAin, :]).transpose(2,0,1)

        self.fluorescence_emitted_intensity = np.nansum(xlf.abs2(self.fluorescence_local_amplitude_propagated),axis=4)

                    
                    
    #@timeit
    def _calc_fluorescence_old(self):

        # We are only interested in active layers
        for n, layer in enumerate(self.cavity.layer_list):
            if not layer.is_active:
                continue
            relevant_z_indices = np.where(self.problem.z_layer_indices == n)[0]
            #LD = self.L_matrices_out
            L1 = self.L_matrices_out_partials#
            L1i = self.L_matrices_out_partials_inverse # Eout,Aout,z(layer)
            L2 = self.L_matrices_out_partials_reverse # Eout,Aout,z(layer)

            R1 =  -L1i[:,:,relevant_z_indices,0, 1] / L1i[:,:,relevant_z_indices,0, 0]# -L1i[:,:,relevant_z_indices,0, 1] / L1i[:,:,relevant_z_indices,0, 0]  # -L1[1,0]/L1[1,1]
            R2 = -L2[:,:,relevant_z_indices,1, 0] / L2[:,:,relevant_z_indices,1, 1]  # -L2[0,1]/L2[1,1]

            excitation_intensity = xlf.abs2(self.incident_field_amplitude[:,:,relevant_z_indices])       # Ein,Ain,z(layer)
            fluorescence_intensity = excitation_intensity * layer.sigma_inel * layer.dz * self.problem.d_angle_out/(2*np.pi) # Ein,Ain,z(layer)
            fluorescence_amplitude = np.sqrt(fluorescence_intensity)   # I Multiply the fluorescence amplitude with the z-resolution to make the result (sum) independent of number of layers.

            #
            #way_to_the_surface = (L1i[:,:,relevant_z_indices,1, 1] - (L1i[:,:,relevant_z_indices,1, 0] * L1i[:,:,relevant_z_indices,0, 1]) / L1i[:,:,relevant_z_indices,0, 0]).transpose(2,0,1) # Eout,Aout, z(layer)
            way_to_the_surface = 1/(L1[:,:,relevant_z_indices,1, 1]).transpose(2,0,1) 
                
            # Non-parallel code
            for iEin, Ein in enumerate(self.problem.energies_in):
                for iAin, Ain in enumerate(self.problem.angles_in):
                    self.fluorescence_local_amplitude[iEin, :,iAin ,:, relevant_z_indices, 0] = ((fluorescence_amplitude[iEin, iAin, :] * R1[...]) / (1 - R2[...] * R1[...])).transpose(2,0,1)  # A+ (down)
                    self.fluorescence_local_amplitude[iEin, :,iAin ,:, relevant_z_indices, 1] = ((fluorescence_amplitude[iEin, iAin, :] * R2[...]) / (1 - R1[...] * R2[...])).transpose(2,0,1)  # A- (up)

                    #A_emitted = (L1i[1, 1] - (L1i[1, 0] * L1i[0, 1]) / L1i[0, 0]) * A_up
                    # Old Version: 
                    self.fluorescence_local_amplitude_propagated[iEin, :,iAin ,:, relevant_z_indices] = way_to_the_surface * self.fluorescence_local_amplitude[iEin, :,iAin ,:, relevant_z_indices, 1]
                    # New Version (16.08.21), reprocisity
                    #self.fluorescence_local_amplitude_propagated[iEin, :,iAin ,:, relevant_z_indices] = ( L1i[:,:,relevant_z_indices,0,0]*self.fluorescence_local_amplitude[iEin, :,iAin ,:, relevant_z_indices, 0].transpose((1,2,0))\
                     #   + L1i[:,:,relevant_z_indices,0,1]*self.fluorescence_local_amplitude[iEin, :,iAin ,:, relevant_z_indices, 1].transpose((1,2,0)) ).transpose(2,0,1)
                        
                    #New version 2
                    #self.fluorescence_local_amplitude_propagated[iEin, :,iAin ,:, relevant_z_indices] = (way_to_the_surface * fluorescence_amplitude[iEin, iAin, :]).transpose(2,0,1)



            """ #This parallel code appears to run longer than the Non-parallel version! Apparently each Chunk is too small, or it might by copying the arrays in between threads...
            # Distribute tasks
            futures_to_results = {}
            def results_for_one_coordinate_pair(iEin, iAin):
                A_up = ((fluorescence_amplitude[iEin, iAin, :] * R2[...]) / (1 - R1[...] * R2[...])).transpose(2,0,1)
                return (((fluorescence_amplitude[iEin, iAin, :] * R1[...]) / (1 - R2[...] * R1[...])).transpose(2,0,1), A_up, way_to_the_surface * A_up)
            for iEin, Ein in enumerate(self.problem.energies_in):
                for iAin, Ain in enumerate(self.problem.angles_in):
                    futures_to_results[self.problem.executor.submit(results_for_one_coordinate_pair, iEin, iAin)] = (iEin, iAin)

            # Collect results
            for future in concurrent.futures.as_completed(futures_to_results):
                (iEin, iAin) = futures_to_results[future]
                try:
                    self.fluorescence_local_amplitude[iEin, :,iAin ,:, relevant_z_indices, 0] = future.result()[0]
                    self.fluorescence_local_amplitude[iEin, :,iAin ,:, relevant_z_indices, 1] = future.result()[1]
                    self.fluorescence_local_amplitude_propagated[iEin, :,iAin ,:, relevant_z_indices] = future.result()[2]
                except Exception as exc:
                    print('%r generated an exception: %s' % (n, exc))
            """
        # Finally, sum all the fluorescence over the z axis
        self.fluorescence_emitted_intensity = np.nansum(self.fluorescence_local_amplitude_propagated,axis=4)


