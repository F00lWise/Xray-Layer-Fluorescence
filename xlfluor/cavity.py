import numpy as np
import lmfit
import xlfluor

DEBUG = True


class Cavity:
    """
    Just a small class to hold the cavity together
    """
    def __init__(self, layer_list):
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

    def propose_problem(self, problem):
        """
        Takes a problem and offers a solution
        :param problem:
        :return:
        """
        self.solution = CavitySolution(self, problem)

    def layer_list_to_parameters(self,layer_list, d_tolerance=1, rho_tolerance=0.5e3):
        """
        takes in a parameter list and generates a layer list from it which can be treated by an optimizer
        :param layer_list:
        :return:
        """
        params = lmfit.Parameters()
        for n, layer in enumerate(layer_list):
            prefix = f'Layer_{n}_{layer.material.name}_'
            if not layer.final:
                params.add(prefix + 'd', value=layer.d, min=layer.d - d_tolerance, max=layer.d + d_tolerance)
            else:
                params.add(prefix + 'd', value=layer.d, vary=False)

            params.add(prefix + 'rho', value=layer.density, min=layer.density - rho_tolerance,
                     max=layer.density + rho_tolerance)
        return params

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
        self.incident_field_amplitude = np.empty((len(problem.energies_in),len(problem.angles_in),len(problem.z_axis)), dtype=complex)*np.nan # scalar at each point in input angle, input energy and z
        self.fluorescence_local_amplitude = np.empty((len(problem.energies_in), len(problem.energies_out),len(problem.angles_in), len(problem.angles_out),len(problem.z_axis), 2), dtype=complex)*np.nan # vector (down, up) at each point
        self.fluorescence_local_amplitude_propagated = np.empty((len(problem.energies_in), len(problem.energies_out),len(problem.angles_in), len(problem.angles_out),len(problem.z_axis)), dtype=complex)*np.nan # fluorescence amplitude from each point in z but propagated to surface
        self.fluorescence_emitted_amplitude = np.empty((len(problem.energies_in), len(problem.energies_out),len(problem.angles_in), len(problem.angles_out)), dtype=complex)*np.nan# scalar (emitted upwards from z=0) at each point in input angle and input energy

        self.layer_solutions = [layer.solve(self.problem) for layer in
                                cavity.layer_list[:-1]]  # calculate initial layer solutions
        self.layer_solutions.append(cavity.layer_list[-1].solve(self.problem))

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

        self._calc_incident_field()

        self._calc_fluorescence()

    def _solve_layers(self, parameters):
        """
        Call all layer.solve() functions with new parameters.
        In the process of layer.solve, the new parameters are also written into the layer parameters
        :param parameters:
        :return:
        """
        for n, layer in enumerate(self.problem.cavity.layer_list):
            prefix = f'Layer_{n}_{layer.material.name}_'
            d = parameters[prefix + 'd'].value
            rho = parameters[prefix + 'rho'].value

            self.layer_solutions[n] = layer.solve(self.problem, d, rho)

    def _calc_L_total(self):
        """
        Calculates the transfer matrix of the entire cavity
        :return:
        """
        self.L_matrices_in[...] = np.eye(2)
        self.L_matrices_out[...] = np.eye(2)
        for n, layer in enumerate(self.cavity.layer_list):
            self.L_matrices_in = layer.solution.L_matrices_in @ self.L_matrices_in
            self.L_matrices_out = layer.solution.L_matrices_out @ self.L_matrices_out

    def _calc_L_partial(self):
        """
        Calculates the transfer matrix of the cavity up to the specific depths in problem.z_axis
        :return:
        """
        self.L_matrices_in_partials[...] = np.eye(2)
        self.L_matrices_out_partials[...] = np.eye(2)

        for iz, z in enumerate(self.problem.z_axis):
            # which layer are we in?
            partial_layer_index: int = self.problem.z_layer_indices[iz]
            partial_layer = self.cavity.layer_list[partial_layer_index]

            # Mask all the values that we do not calculate anyways
            if not partial_layer.is_active:
                self.L_matrices_out_partials[:, :, iz, :, :] = np.nan
                self.L_matrices_out_partials_inverse[:, :, iz, :, :] = np.nan


            # Cavity Matrix up to layer in question
            for n, layer in enumerate(self.cavity.layer_list[:partial_layer_index]):
                self.L_matrices_in_partials[:,:,iz,:,:] = layer.solution.L_matrices_in[:,:,:,:]  @ self.L_matrices_in_partials[:,:,iz,:,:]
                if partial_layer.is_active:
                    self.L_matrices_out_partials[:,:,iz,:,:] = layer.solution.L_matrices_out[:,:,:,:] @ self.L_matrices_out_partials[:,:,iz,:,:]

            # Matrix of partial layer
            iz_within_layer = iz - np.where(self.problem.z_layer_indices == partial_layer_index)[0][0] # where statement finds the first z-index of the current partial layer
            self.L_matrices_in_partials[:,:,iz_within_layer,:,:] = partial_layer.solution.L_matrices_in_partials[:,:,iz_within_layer,:,:]  @ self.L_matrices_in_partials[:,:,iz,:,:]
            if partial_layer.is_active:
                self.L_matrices_out_partials[:,:,iz_within_layer,:,:] = partial_layer.solution.L_matrices_out[:,:,:,:] @ self.L_matrices_out_partials[:,:,iz,:,:]

            # calculate reverse matrices: This equates to: L_2 = L_D @ L_1^-1
            if partial_layer.is_active:
                self.L_matrices_out_partials_inverse[:,:,iz,:,:] = np.linalg.inv(self.L_matrices_out_partials[:,:,iz,:,:]) #L(z_p)^-1
                self.L_matrices_out_partials_reverse[:,:,iz,:,:] = self.L_matrices_out @ self.L_matrices_out_partials_inverse[:,:,iz,:,:] # L2 = L(D) @ L ^-1

    def consistency_check(self):
        Assembled_L = self.L_matrices_out_partials_reverse @ self.L_matrices_out_partials

        imprecision = np.empty(self.L_matrices_out_partials_reverse.shape)
        for iz,z in self.problem.z_axis:
            imprecision[:,:,iz,:,:] = Assembled_L - self.L_matrices_out

    def calc_R(self):
        """
        Cavity reflectivity of incident field according to last solve() call
        :return:
        """
        return -(self.L_matrices_in[:,:,1, 0] / self.L_matrices_in[:,:,1, 1])

    def calc_T(self):
        """
        Cavity transmission of incident field according to last solve() call
        :return:
        """
        return self.L_matrices_in[0, 0] - (self.L_matrices_in[0, 1] * self.L_matrices_in[1, 0]) / self.L_matrices_in[1, 1]

    def _calc_incident_field(self):
        """
        Calculate the field strength of the incident wave at all depth in z_axis
        :return:
        """
        for iz,z in enumerate(self.problem.z_axis):
            self.incident_field_amplitude[:, :, iz] = self.L_matrices_in_partials[:, :, iz, 0, 0] + self.L_matrices_in_partials[:, :, iz, 1, 0] - \
                                                      (self.L_matrices_in_partials[:,:,iz,0,1] + self.L_matrices_in_partials[:,:,iz,1,1]) * \
            self.L_matrices_in[:,:,1,0] / self.L_matrices_in[:,:,1,1]

    def _calc_fluorescence(self):

        # We are only interested in active layers
        for n, layer in enumerate(self.cavity.layer_list):
            if not layer.is_active:
                continue
            relevant_z_indices = np.where(self.problem.z_layer_indices == n)[0]
            #LD = self.L_matrices_out
            #L1 = self.L_matrices_out_partials#
            L1i = self.L_matrices_out_partials_inverse # Eout,Aout,z(layer)
            L2 = self.L_matrices_out_partials_reverse # Eout,Aout,z(layer)

            R1 = -L1i[:,:,relevant_z_indices,0, 1] / L1i[:,:,relevant_z_indices,0, 0]  # -L1[1,0]/L1[1,1]
            R2 = -L2[:,:,relevant_z_indices,1, 0] / L2[:,:,relevant_z_indices,1, 1]  # -L2[0,1]/L2[1,1]

            excitation_intensity = xlfluor.abs2(self.incident_field_amplitude[:,:,relevant_z_indices])       # Ein,Ain,z(layer)
            fluorescence_intensity = excitation_intensity * layer.sigma_inel * layer.dz * self.problem.d_angle_out/(2*np.pi) # Ein,Ain,z(layer)
            fluorescence_amplitude = np.sqrt(fluorescence_intensity)   # I Multiply the fluorescence amplitude with the z-resolution to make the result (sum) independent of number of layers.

            way_to_the_surface = (L1i[:,:,relevant_z_indices,1, 1] - (L1i[:,:,relevant_z_indices,1, 0] * L1i[:,:,relevant_z_indices,0, 1]) / L1i[:,:,relevant_z_indices,0, 0]).transpose(2,0,1) # Eout,Aout, z(layer)
# Good point to parallelize
            for iEin, Ein in enumerate(self.problem.energies_in):
                for iAin, Ain in enumerate(self.problem.angles_in):
                    self.fluorescence_local_amplitude[iEin, :,iAin ,:, relevant_z_indices, 0] = ((fluorescence_amplitude[iEin, iAin, :] * R1[...]) / (1 - R2[...] * R1[...])).transpose(2,0,1)  # A+ (down)
                    self.fluorescence_local_amplitude[iEin, :,iAin ,:, relevant_z_indices, 1] = ((fluorescence_amplitude[iEin, iAin, :] * R2[...]) / (1 - R1[...] * R2[...])).transpose(2,0,1)  # A- (up)

                    #A_emitted = (L1i[1, 1] - (L1i[1, 0] * L1i[0, 1]) / L1i[0, 0]) * A_up
                    self.fluorescence_local_amplitude_propagated[iEin, :,iAin ,:, relevant_z_indices] = way_to_the_surface * self.fluorescence_local_amplitude[iEin, :,iAin ,:, relevant_z_indices, 1]

        # Finally, sum all the fluorescence over the z axis
        self.fluorescence_emitted_amplitude = np.nansum(self.fluorescence_local_amplitude_propagated,axis=4)
