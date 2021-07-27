import numpy as np

class Cavity:
    def __init__(self, layer_list):
        self.D = np.sum([layer.d for layer in layer_list[:-1]])
        self.layer_list = layer_list

    def L(self, E, theta, z=None):
        if z is not None and (z * 1e-9 < self.D):
            return self._L_partial(E, theta, z * 1e-9)  # z: nm->m

        L_center = np.identity(2)
        for layer in self.layer_list[:-1]:
            L_center = layer.L(E, theta) @ L_center

        L_substrate = self.layer_list[-1].L_final(E, theta)
        L = L_substrate @ L_center
        return L

    def _L_partial(self, E, theta, z):
        d_represented = 0
        reached_z = False
        L_center = np.identity(2)
        for i, layer in enumerate(self.layer_list[:-1]):
            if d_represented + layer.d > z:
                L_center = layer.L(E, theta, (d_represented + layer.d - z)) @ L_center
                reached_z = True
                break
            else:
                L_center = layer.L(E, theta) @ L_center
                d_represented += layer.d
        if not reached_z:
            print('Warning: I did not reach the prompted depdth with the known layers')
            return np.nan
        return L_center

    def R(self, E, theta):
        L = self.L(E, theta)
        return -(L[1, 0] / L[1, 1])

    def T(self, E, theta):
        L = self.L(E, theta)
        return L[0, 0] - (L[0, 1] * L[1, 0]) / L[1, 1]

    def field(self, E, theta, z):
        Lz = self.L(E, theta, z)
        LD = self.L(E, theta)

        a = Lz[0, 0] + Lz[1, 0] - (Lz[0, 1] + Lz[1, 1]) * LD[1, 0] / LD[1, 1]
        return a

    def fluor(self, E, theta, z, field_str, sigma_inel):
        # Calculate partial transmission matrizes
        LD = self.L(E, theta)
        L1 = self.L(E, theta, z)
        L1i = np.linalg.inv(L1)
        L2 = (LD @ L1i)  #

        # This works now, so I save the computation time
        if not np.all(LD - L2 @ L1 < 1e-5):
            print(f'Waarning!!! Consistency check failed by {LD - L2 @ L1}\n theta = {lh.rad2deg(theta)}°')

        R1 = -L1i[0, 1] / L1i[0, 0]  # -L1[1,0]/L1[1,1]
        R2 = -L2[1, 0] / L2[1, 1]  # -L2[0,1]/L2[1,1]

        excitation_intensity = np.real(field_str * np.conj(field_str))
        # randomize fluorescence phase?
        fluorescence_amplitude = np.sqrt(sigma_inel * excitation_intensity)  # *np.exp(1j*np.pi*2*np.random.random())

        A_down = (fluorescence_amplitude * R1) / (1 - R2 * R1)  # A+
        A_up = (fluorescence_amplitude * R2) / (1 - R1 * R2)  # A-
        A_local = np.array([A_down, A_up]).T

        # print(f'a={excitation_amplitude}, R1,2 = {cplxsq(R1)},{cplxsq(R2)} => A = ({cplxsq(A_up)},{cplxsq(A_down)})')

        # A_emitted_old = L1i@A_local
        # Direct derivation using A0_down = 0
        A_emitted = (L1i[1, 1] - (L1i[1, 0] * L1i[0, 1]) / L1i[0, 0]) * A_up

        # print(A_emitted, A_emitted_old)
        return A_emitted, A_local

