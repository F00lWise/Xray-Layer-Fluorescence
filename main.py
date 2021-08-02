import xlfluor as xlf
import numpy as np
import matplotlib.pyplot as plt
import lmfit
import matplotlib as mpl

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


#### Let us read some experimental data for comparison
keys = ['dt', 'dtz', 'fluor_diode', 'izero_diode', 'refl', 'scanNr',
       'sry', 'sty']
loaded_scan_mat = np.loadtxt('scan_486_wide_angle_diode.txt')
loaded_scan = {}
for i, key in enumerate(keys):
    loaded_scan[key] = loaded_scan_mat[1:,i]

experiment_data = {
    'fluor_diode': xlf.normmax(loaded_scan['fluor_diode']),
    'refl': xlf.normmax(loaded_scan['refl']),
    'angles_in': xlf.deg2rad(np.array(loaded_scan['sry'])),
    'energies_out': np.array([6400])
}


#### Manual solution for testing
energies_in=np.array([7150])
energies_out=np.array([6400])
angles_in  = np.linspace(xlf.deg2rad(0.2),xlf.deg2rad(1.0),100) #np.array([xlf.deg2rad(0.3)])#
angles_out = np.linspace(xlf.deg2rad(0.2),xlf.deg2rad(1.0),100) #np.array(xlf.deg2rad(np.array([0.2,0.3])))#
#angles_in  = np.array([lh.deg2rad(0.3)])#np.linspace(lh.deg2rad(0.2),lh.deg2rad(0.6),40)
#angles_out = np.array(lh.deg2rad(np.array([0.2,0.3])))#np.linspace(lh.deg2rad(0.2),lh.deg2rad(1.0),40)
axes = {'energies_in': energies_in,
        'energies_out':energies_out,
        'angles_in': angles_in,
        'angles_out':angles_out}


my_problem = xlf.Problem(cavity, experiment_data = experiment_data, axes=axes, passive_layer_resolution = 1, active_layer_resolution = 10)


parameters = cavity.parameters

my_problem.solve(cavity, parameters)

cavity.solution.consistency_check()




#############################################
######### Final Plotting



###################################################
##### Diode trace plots
###################################################
angles_in = xlf.rad2deg(my_problem.angles_in)
model_fluor = xlf.abs2(my_problem.fluorescence_I_angle_in_dependent)
model_refl = xlf.abs2(my_problem.reflectivity)[0, :]

exp_fluor = my_problem.experiment['fluor_diode'] * parameters['fluorescence_scaling'].value
exp_refl = my_problem.experiment['refl'] * parameters['reflectivity_scaling'].value
angle_error = parameters['experiment_angle_err'].value

exp_refl = xlf.shift_by_delta(x=angles_in, sft=angle_error, y=exp_refl, oversampling=10)
exp_fluor = xlf.shift_by_delta(x=angles_in, sft=angle_error, y=exp_fluor, oversampling=10)


fig, (ax1,ax2) = plt.subplots(2,1,figsize=(7, 5))
plt.sca(ax1)
ax1.plot(angles_in, exp_refl, c='k', lw=2,label = 'experiment')
ax1.plot(angles_in, model_refl, 'C4-', label='Simulated refl')

plt.ylabel('Normalized Intensity / arb. u.')
plt.xlabel('Input Angle / °')
plt.ylabel('Fluorescence')
plt.legend()

plt.sca(ax2)
ax2.plot(angles_in, exp_fluor, c='k', lw=2,label = 'experiment')
ax2.plot(angles_in, model_fluor, c='C3',
         label='Simulated Fluorescence')
plt.xlabel('Input Angle / °')
plt.suptitle(f'Input Angle Dependencies')
plt.legend()
ax2 = plt.gca().twinx()
plt.ylabel('Reflectivity')
#plt.ylim(-1, None)
#plt.yticks([0, 0.5, 1])
plt.axhline(c='k', lw=.5)

"""
mode_angles = [.253, .2725, .317, .3625, .420, .474, .536, .598, .662]
for angle in mode_angles:
    plt.axvline(angle - data_shift, lw=.5, c='C0', ls='--')
mode_angles = [.253, .276, .318, .365, .420, .482, .536, .606, .655]
for angle in mode_angles:
    plt.axvline(angle - data_shift, lw=.5, c='C1', ls='--')
"""






###########################################
##### Optimization

iteration_counter = xlf.optimization.counter()

## Prepare plot
fig, (ax1,ax2, ax3) = plt.subplots(3,1,figsize=(7, 5))
plt.sca(ax1)
plt.ylabel('Normalized Intensity / arb. u.')
plt.xlabel('Input Angle / °')
plt.ylabel('Fluorescence')
plt.sca(ax2)
plt.xlabel('Input Angle / °')
plt.suptitle(f'Input Angle Dependencies')

plt.sca(ax3)
plt.xlabel('# Iterations')
ax3.set_ylabel('Reflectivity Residual')
plt.suptitle(f'Residual Plot')
ax4 = ax3.twinx()
ax4.set_ylabel('Fluorescence Residual')

#minimize
minimizer = lmfit.Minimizer(xlf.optimization.cost_function, params=parameters, fcn_args=(my_problem, iteration_counter, (ax1,ax2, ax3,ax4)), iter_cb = xlf.optimization.fit_monitoring)#

result = minimizer.minimize(method = 'leastsq')

lmfit.report_fit(result)


"""
#####################################################
######### 2d Plot
################################################
plotmat = xlf.abs2(cavity.solution.fluorescence_emitted_amplitude[0,0,:,:])
#plotmat = xlf.abs2(np.nansum(cavity.solution.fluorescence_local_amplitude[0,0,:,:,:,0],2)) # sum over depth of non-propagated fluorescence
#plotmat = cplxsq(my_problem.fluor_emitted_from_z[:,:,:])
#plotmat = cplxsq(np.sum(my_problem.fluor_emitted_from_z[:,:,:],0))

plt.figure()
plt.pcolormesh(xlf.rad2deg(my_problem.angles_in),xlf.rad2deg(my_problem.angles_out),\
               plotmat.T, cmap = 'gnuplot', shading = 'nearest')#,vmax=4e3)#norm = mpl.colors.LogNorm(vmin = 0.005),
plt.ylabel('Output angle / °')
plt.xlabel('Input angle / °')
plt.title('Emitted Fluorescence Intensity')
plt.colorbar(label=r'$\tilde{I} / I_0$')
plt.tight_layout()


#########################################


example_angle_in = xlf.deg2rad(0.3)
example_angle_in_index = np.argmin(np.abs(example_angle_in-my_problem.angles_in))

example_angle_out = xlf.deg2rad(0.342)
example_angle_out_index = np.argmin(np.abs(example_angle_out-my_problem.angles_out))



fig, axes = plt.subplots(2,1, figsize = (7,5))
plt.sca(axes[0])
plt.pcolormesh(xlf.rad2deg(my_problem.angles_in),my_problem.z_axis*1e9,\
               xlf.abs2(cavity.solution.fluorescence_local_amplitude_propagated[0,0,:,example_angle_out_index,:]).T,shading = 'nearest',cmap = 'gnuplot')
plt.axvline(xlf.rad2deg(example_angle_in), ls = '--', lw= 1, c='grey')

plt.ylabel('Cavity depth $z_p$ / nm')
plt.xlabel('Input angle / °')
plt.title(r'Fluorescence Intensity $\theta_{in}$ dependency at $\theta_{out}$ ='+f' {xlf.rad2deg(my_problem.angles_out[example_angle_out_index]):.2}°')
plt.colorbar(label=r'$\tilde{I}_{-} / I_0$')
plt.gca().invert_yaxis()
plt.ylim(32.5,36.5)
plt.tight_layout()
plt.gca().invert_yaxis()

plt.sca(axes[1])
plt.pcolormesh(xlf.rad2deg(my_problem.angles_out),my_problem.z_axis*1e9,\
               xlf.abs2(cavity.solution.fluorescence_local_amplitude_propagated[0,0,example_angle_in_index,:,:]).T,shading = 'nearest', vmax=None,cmap = 'gnuplot')#, norm = mpl.colors.LogNorm()
plt.axvline(xlf.rad2deg(example_angle_out), ls = '--', lw= 1, c='grey')
plt.ylabel('Cavity depth $z_p$ / nm')
plt.xlabel('Output angle / $^\circ$')
plt.title(r'Fluorescence Intensity $\theta_{out}$ dependency at $\theta_{in}$ ='+f' {xlf.rad2deg(my_problem.angles_in[example_angle_in_index]):.2}°')
plt.colorbar(label=r'$\tilde{I}_{-} / I_0$')
plt.gca().invert_yaxis()
plt.ylim(32.5,36.5)
plt.tight_layout()
plt.gca().invert_yaxis()

###################################

plt.figure(figsize = (7,5))
plt.pcolormesh(xlf.rad2deg(my_problem.angles_in),my_problem.z_axis*1e9, xlf.abs2(cavity.solution.incident_field_amplitude[0,:,:].T),\
               shading = 'nearest',cmap = 'gnuplot')
for layer in cavity.layer_list:
    plt.axhline(layer.min_z*1e9)
plt.ylabel('Cavity depth $z$ / nm')
plt.xlabel('Incident angle $ { \Theta}_{in}$ / $^\circ$')
plt.title('Excitation Intensity')
#plt.colorbar(label='$I / I_0$')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.ylim(80,None)
"""
plt.show()

my_problem.__del__()