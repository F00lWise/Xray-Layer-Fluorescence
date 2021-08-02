import lmfit
import matplotlib.pyplot as plt
import xlfluor as xlf
import numpy as np


def counter():
    i = 0
    while True:
        yield i
        i = i + 1


def cost_function(parameters, problem, iteration_counter, plot_axes = None):
    iteration = next(iteration_counter)
    print(f'Fitting with parameters:')
    print(parameters)

    problem.solve(problem.cavity,parameters)

    angles_in = xlf.rad2deg(problem.angles_in)
    model_fluor = xlf.abs2(problem.fluorescence_I_angle_in_dependent)
    model_refl = xlf.abs2(problem.reflectivity)[0, :]

    exp_fluor =problem.experiment['fluor_diode'] * parameters['fluorescence_scaling'].value
    exp_refl = problem.experiment['refl']* parameters['reflectivity_scaling'].value
    angle_error = parameters['experiment_angle_err'].value

    exp_refl = xlf.shift_by_delta(x = angles_in, sft = angle_error, y = exp_refl, oversampling= 10, mask_extrapolated_data= True)
    exp_fluor = xlf.shift_by_delta(x = angles_in, sft = angle_error, y = exp_fluor, oversampling= 10, mask_extrapolated_data= True)

    residual_refl = (exp_refl - model_refl ) ** 2
    residual_fluor = (exp_fluor - model_fluor ) ** 2

    # Nan Handling: I set the extrapolated values to the average residual
    bad = np.isnan(residual_refl) | np.isnan(residual_fluor)
    good = ~bad
    print(f'Bad points: {np.sum(bad)}')
    residual_refl[bad] = np.mean(residual_refl[good])
    residual_fluor[bad] = np.mean(residual_fluor[good])

    residual = np.concatenate( [residual_fluor[good] * parameters['reflectivity_scaling'].value,  residual_refl[good] * parameters['fluorescence_scaling'].value])

    plot_every_N = 10

    if plot_axes is not None and np.mod(iteration, plot_every_N) == 0:
        color = f'C{np.floor_divide(iteration, plot_every_N)}'
        if iteration == 0:
            plot_axes[0].plot(angles_in[good], exp_refl[good],c='k',lw=2)
            plot_axes[1].plot(angles_in[good], exp_fluor[good],c='k',lw=2)

        plot_axes[0].plot(angles_in[good], model_refl[good], c=color)
        plot_axes[0].plot(angles_in, residual_refl, 'x', c=color)

        plot_axes[1].plot(angles_in[good], model_fluor[good], c=color)
        plot_axes[1].plot(angles_in, residual_fluor, '.', c=color)



        plt.show(block=False)
        plt.pause((0.1))

    return residual


def fit_monitoring(parameters, iter, residual, problem, iteration_counter, plot_axes = None):
    L = len(problem.angles_in)
    total_refl_residual = np.sum(residual[:L])
    total_fluor_residual = np.sum(residual[L:])
    #print(total_refl_residual,total_fluor_residual)

    plot_axes[2].plot(iter, total_refl_residual, 'x',c = 'C0')
    plot_axes[3].plot(iter, total_fluor_residual, 'o',c = 'C1')
    plt.show(block=False)
    plt.pause((0.1))