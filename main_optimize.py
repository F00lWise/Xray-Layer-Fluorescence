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
    xlf.Layer(platinum, 1.755 * 1e-9, density=20.5e3),
    xlf.Layer(carbon, 31.26 * 1e-9, density=1.656e3),
    xlf.Layer(iron_oxide, 3.9 * 1e-9, inelastic_cross=0.4e9, density=4.309e3),
    xlf.Layer(carbon, 34.136 * 1e-9, density=1.589e3),
    xlf.Layer(platinum, 15 * 1e-9, density=20.50e3),
    xlf.Layer(silicon, np.inf, density = 2.614*1e3, final = True)
]


cavity = xlf.Cavity(layer_list_custom)

#### Let us read some experimental data to fit to
keys = ['dt', 'dtz', 'fluor_diode', 'izero_diode', 'refl', 'scanNr',
       'sry', 'sty']
loaded_scan_mat = np.loadtxt('scan_486_wide_angle_diode.txt')
loaded_scan = {}
for i, key in enumerate(keys):
    loaded_scan[key] = loaded_scan_mat[1:,i]

experiment_data = {
    'fluor_diode': xlf.normmax(loaded_scan['fluor_diode']),
    'refl': xlf.normmax(loaded_scan['refl']),
    'angles_in': np.array(loaded_scan['sry']),
    'energies_out': np.array([6400])
}

##### Accounting for experimental offset in sry calibration
data_shift = 0.025
# In this dataset it appears that the "sry" motor was offset by 0.025°
experiment_data['angles_in'] = xlf.deg2rad(experiment_data['angles_in'] - data_shift)


#### Define coordinate space adding to dimensions given by the experiment
energies_in=np.array([7150])
energies_out=np.array([6400])
angles_in  = np.linspace(xlf.deg2rad(0.1),xlf.deg2rad(1.0),200)
angles_out = np.linspace(xlf.deg2rad(0.1),xlf.deg2rad(1.0),3)

axes = {'energies_in': energies_in,
        'energies_out':energies_out,
        'angles_in': angles_in,
        'angles_out':angles_out}


### Define the problem and calculate initial solution
my_problem = xlf.Problem(cavity, experiment_data = experiment_data, axes=axes, passive_layer_resolution = 30, active_layer_resolution = 30)

parameters = cavity.parameters

my_problem.solve(cavity, parameters)

cavity.solution.consistency_check()



############################################
######### Optimization Part

# Get the relative intensities of fluorescence and reflectivity
cavity.get_relative_intensities(my_problem)

# Set relative fitting weights
cavity.set_fit_weigths(weight_refl = 1, weight_fluor = 2)


# Set up the logger
logger = xlf.FitLogger(my_problem, parameters, maxiter = 12, intermediate_plotting = 3)

# Run the optimization

minimizer = lmfit.Minimizer(xlf.cost_function, params=parameters,\
                            fcn_args= (my_problem,),\
                            iter_cb = logger.logging)#xlf.fit_monitoring

result = minimizer.minimize(method = 'leastsq')#

lmfit.report_fit(result)

#### Plot results
logger.final_plot()

plt.show(block = True)