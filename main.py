import xlfluor as xlf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from FLASHutil import little_helpers as lh

global DEBUG
DEBUG = True

# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

#### Let us build a cavity
vacuum = xlf.Vacuum()
carbon = xlf.Element(name = 'c', Z= 6, atom_weight = 12.011,density = 2.2*1e3) # the factor 1e3 converts from g/cm3 to Kg/m3
silicon = xlf.Element(name = 'si', Z= 14, atom_weight = 28.086,density = 2.33*1e3)
iron = xlf.Element(name = 'fe', Z = 26, atom_weight=55.845, density=7.87*1e3)
oxygen = xlf.Element(name = 'o', Z = 8, atom_weight=15.999, density=0.143E-02*1e3) # Oxygen density is as gas!!
platinum = xlf.Element(name = 'pt', Z = 78, atom_weight=195.078, density=21.5*1e3)

iron_oxide = xlf.Composite(name = 'fe2o3', elements=[iron, oxygen], density=5.07*1e3, composition =[2,3])

layer_list_custom = [
    xlf.Layer(platinum, 2.8 * 1e-9, density=21.0e3),
    xlf.Layer(carbon, 31.26 * 1e-9, density=1.7e3),
    xlf.Layer(iron_oxide, 3.9 * 1e-9, inelastic_cross=0.4e9, density=4.5e3),
    xlf.Layer(carbon, 34.136 * 1e-9, density=1.6e3),
    xlf.Layer(platinum, 15 * 1e-9, density=21.00e3),
    xlf.Layer(silicon, np.inf, density = 2.33*1e3, final = True)
]

cavity = xlf.Cavity(layer_list_custom)

parameters = cavity.parameters

"""#### Let us read some experimental data for comparison
wide_scan = pd.read_pickle('scan_486_wide_angle_diode.gz')
experiment_data = {
    'fluor_trace': lh.normmax(wide_scan.fluor_diode),
    'refl_trace': lh.normmax(wide_scan.refl),
    'angles_in': np.array(wide_scan['sry']),
    'angles_out': np.array(wide_scan['sry']),
    'energies_in': np.array([wide_scan['energy'], ]),
    'energies_out': np.array([6400, ])
}
"""

#### Manual solution for testing
energies_in=np.array([7300])
energies_out=np.array([6400])
angles_in = np.linspace(lh.deg2rad(0.2),lh.deg2rad(1.0),200)
angles_out =np.linspace(lh.deg2rad(0.2),lh.deg2rad(1.0),200)

axes = (energies_in,energies_out,angles_in,angles_out)


my_problem = xlf.Problem(cavity, experiment_data = None, axes=axes, passive_layer_resolution = 1, active_layer_resolution = 10)

my_problem.solve(cavity, parameters)


plt.figure(figsize=(7, 5))
ax1 = plt.gca()
plt.ylabel('Normalized Intensity / arb. u.')
plt.plot(lh.rad2deg(my_problem.angles_in), xlf.abs2(my_problem.reflectivity)[0,:] * np.nan, 'C4-',
         label='Simulated Reflectivity')  # dummy plots for legend
plt.plot(lh.rad2deg(my_problem.angles_in), xlf.abs2(my_problem.reflectivity)[0,:] * np.nan, 'C0--',
         label='Measured Reflectivity')

plt.plot(lh.rad2deg(my_problem.angles_in), lh.normmax(xlf.abs2(my_problem.fluorescence_I_angle_in_dependent)), c='C3',
         label='Simulated Fluorescence')


data_shift = 0

# plt.ylim(None,6)
plt.xlabel('Input Angle / °')
plt.title(f'Input Angle Dependencies')
plt.yticks([])
plt.ylim(None, 2)

ax2 = plt.gca().twinx()
plt.plot(lh.rad2deg(my_problem.angles_in), lh.normmax(xlf.abs2(my_problem.reflectivity)[0,:]), 'C4-', label='Simulated refl')
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
