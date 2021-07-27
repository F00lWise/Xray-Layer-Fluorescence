import xlfluor as xlf
import numpy as np
import matplotlib.pyplot as plt

from FLASHutil import little_helpers as lh


# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


vacuum = xlf.Vacuum()
carbon = xlf.Element(name = 'c', Z= 6, atom_weight = 12.011,density = 2.2*1e3) # the factor 1e3 converts from g/cm3 to Kg/m3
silicon = xlf.Element(name = 'si', Z= 14, atom_weight = 28.086,density = 2.33*1e3)
iron = xlf.Element(name = 'fe', Z = 26, atom_weight=55.845, density=7.87*1e3)
oxygen = xlf.Element(name = 'o', Z = 8, atom_weight=15.999, density=0.143E-02*1e3) # Oxygen density is as gas!!
platinum = xlf.Element(name = 'pt', Z = 78, atom_weight=195.078, density=21.5*1e3)

iron_oxide = xlf.Composite(name = 'fe2o3', elements=[iron, oxygen],density=5.07*1e3,composition =[2,3])

layer_list_custom2 = [
    xlf.Layer(platinum,2.8,depth_resolution=1, density=21.0e3),
    xlf.Layer(carbon,31.26,depth_resolution=1, density=1.7e3),
    xlf.Layer(iron_oxide,3.9, depth_resolution=15, inelastic_cross=0.4, density=4.5e3),
    xlf.Layer(carbon,34.136, depth_resolution=1, density=1.6e3),
    xlf.Layer(platinum,15, depth_resolution=1, density=21.00e3),
    xlf.Layer(silicon,np.inf)
]
cavity = xlf.Cavity(layer_list_custom2)



if False:
    #### Manual solution for testing
    energy_in=7300
    energy_out=6400

    angle_in = lh.deg2rad(0.5)
    angles_out=np.linspace(lh.deg2rad(0.2),lh.deg2rad(1.0),240)

    z = 35

    incident_field = cavity.field(energy_in,angle_in,z)

    fluor_emitted_from_z = np.zeros((len(angles_out),1),dtype=np.complex)
    fluor_local_amplitude = np.zeros((len(angles_out),2),dtype=np.complex)


    for iao, angle_out in enumerate(angles_out):
        fluor_emitted_from_z[iao], fluor_local_amplitude[iao] = \
                cavity.fluor(energy_out,\
                              angle_out,\
                              z,\
                              incident_field,\
                              0.1)

    plt.figure()

    plt.plot(lh.rad2deg(angles_out),xlf.abs2(fluor_emitted_from_z[:]),'.-')

    plt.show(block = False)


my_problem = xlf.Problem(cavity, energy_in=7150, energy_out=6400,\
                     angles_in=np.linspace(lh.deg2rad(0.2),lh.deg2rad(0.6),3),\
                     angles_out=np.linspace(lh.deg2rad(0.2),lh.deg2rad(1.0),3))

my_problem.solve()


data_shift = 0.025
plt.figure(figsize=(7, 5))
ax1 = plt.gca()

plt.ylabel('Normalized Intensity / arb. u.')
plt.plot(lh.rad2deg(my_problem.angles_in),abs2(my_problem.reflectivity)*np.nan,'C4-', label = 'Simulated Reflectivity') # dummy plots for legend
plt.plot(lh.rad2deg(my_problem.angles_in),abs2(my_problem.reflectivity)*np.nan,'C0--', label = 'Measured Reflectivity')

plt.plot(lh.rad2deg(my_problem.angles_in),lh.normmax(my_problem.fluotescence_I_angle_in_dependent),c='C3', label = 'Simulated Fluorescence')


# plt.ylim(None,6)
plt.xlabel('Input Angle / °')
plt.title(f'Input Angle Dependencies')
plt.yticks([])
plt.ylim(None, 2)

ax2 = plt.gca().twinx()
plt.plot(lh.rad2deg(my_problem.angles_in), lh.normmax(xlf.abs2(my_problem.reflectivity)), 'C4-', label='Simulated refl')
plt.ylabel('Reflectivity')
plt.ylim(-1, None)
plt.yticks([0, 0.5, 1])
plt.axhline(c='k', lw=.5)

mode_angles = [.253, .2725, .317, .3625, .420, .474, .536, .598, .662]
for angle in mode_angles:
    plt.axvline(angle - data_shift, lw=.5, c='C0', ls='--')
mode_angles = [.253, .276, .318, .365, .420, .482, .536, .606, .655]
for angle in mode_angles:
    plt.axvline(angle - data_shift, lw=.5, c='C1', ls='--')

plt.show()
