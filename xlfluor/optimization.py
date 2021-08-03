import lmfit
import matplotlib.pyplot as plt
import xlfluor as xlf
import numpy as np
import pandas as pd


def counter():
    i = 0
    while True:
        yield i
        i = i + 1

def norm(*args):
    """
    This is the function applied as a norm for the fit.
    
    L1 norm would be np.abs(*args)
    L2 norm would be np.square(*args)
    
    """
    #return np.square(*args)
    return np.abs(*args)

def cost_function(parameters, problem):
    #print(f'Fitting with parameters:')
    #print(parameters)
    problem.cavity.parameters = parameters
    
    problem.solve(problem.cavity,parameters)

    angles_in = xlf.rad2deg(problem.angles_in)
    model_fluor = xlf.abs2(problem.fluorescence_I_angle_in_dependent)/ parameters['I_fluorescence'].value
    model_refl = np.mean(xlf.abs2(problem.reflectivity),0) / parameters['I_reflectivity'].value

    residual_refl = norm((problem.experiment['refl'] - model_refl ))
    residual_fluor = norm((problem.experiment['fluor_diode'] - model_fluor))

    residual = np.concatenate( [residual_fluor * parameters['weight_fluorescence'].value,\
                                residual_refl *  parameters['weight_reflectivity'].value])

    return residual


def fit_monitoring(parameters, iteration_counter, residual, problem, plot_axes = None):
    print(f'Iteration {iteration_counter}')
    angles_in = xlf.rad2deg(problem.angles_in)
    model_fluor = xlf.abs2(problem.fluorescence_I_angle_in_dependent)/ parameters['I_fluorescence'].value
    model_refl = np.mean(xlf.abs2(problem.reflectivity),0) / parameters['I_reflectivity'].value

    L = len(angles_in)
    residual_fluor = residual[:L]
    residual_refl =residual[L:]
    total_refl_residual = np.sum(residual_fluor)
    total_fluor_residual = np.sum(residual_refl)

    plot_every_N = 10

    if plot_axes is not None and np.mod(iteration_counter, plot_every_N) == 0:
        color = f'C{np.floor_divide(iteration_counter, plot_every_N)}'
        if iteration_counter == 0:
            plot_axes[0].plot(angles_in, problem.experiment['refl'],c='k',lw=2)
            plot_axes[1].plot(angles_in, problem.experiment['fluor_diode'],c='k',lw=2)

        plot_axes[0].plot(angles_in, model_refl, c=color)
        plot_axes[0].plot(angles_in, residual_refl, 'x', c=color)

        plot_axes[1].plot(angles_in, model_fluor, c=color)
        plot_axes[1].plot(angles_in, residual_fluor, '.', c=color)
        
        plt.show(block=False)
        plt.pause(0.1)

    plot_axes[2].plot(iteration_counter, total_refl_residual, 'x',c = 'C0')
    plot_axes[2].plot(iteration_counter, total_fluor_residual, 'o',c = 'C1')
    
class FitLogger:
    def __init__(self, problem, parameters, maxiter = 1000):
        
        self.parameters = parameters
        self.problem = problem
        self.maxiter = maxiter
        
        # Make a pandas series from the parameter values
        s_par = pd.Series({par: parameters[par].value for par in parameters}, name = -1)
        
        # And start a dataset with them
        self.df_par = pd.DataFrame(columns = [par for par in parameters])
        self.df_par = self.df_par.append(s_par)
        
        # Compute initial residuals
        model_fluor = xlf.abs2(problem.fluorescence_I_angle_in_dependent)/ parameters['I_fluorescence'].value
        model_refl = np.mean(xlf.abs2(problem.reflectivity),0) / parameters['I_reflectivity'].value
        residual_refl = (xlf.norm(problem.experiment['refl'] - model_refl ))  * parameters['weight_reflectivity'].value
        residual_fluor = (xlf.norm(problem.experiment['fluor_diode'] - model_fluor))  * parameters['weight_fluorescence'].value

        s_resid = pd.Series({'Refl':np.sum(residual_refl),'Fluor':np.sum(residual_fluor)}, name = -1)
        
        # And start a dataset with them
        self.df_resid = pd.DataFrame(columns = ['Refl', 'Fluor'])
        self.df_resid = self.df_resid.append(s_resid)
        
        
        ##### Initial solution plot
        self.fitfig, self.fitaxes  = plt.subplots(2,1,figsize=(7, 5))
        angles_in = xlf.rad2deg(problem.angles_in)

        ## Reflectivity
        plt.sca(self.fitaxes[0])
        plt.title('Data as fitted')
        plt.xlabel('Input Angle / °')
        plt.ylabel('Reflectivity')
        
        exp_refl = problem.experiment['refl']
        plt.plot(angles_in, exp_refl, c='red', lw=2,label = 'Experiment')
        plt.plot(angles_in, model_refl, 'C0-', label='Initial Fit')
        
        self.fitaxes_residual0 = self.fitaxes[0].twinx()
        self.fitaxes_residual0.plot(angles_in,residual_refl, '.', c = 'grey')
        plt.ylabel('Residual')
        
        
        ## Fluorescence
        plt.sca(self.fitaxes[1])
        plt.xlabel('Input Angle / °')
        plt.ylabel('Fluorescence')
        
        exp_fluor = self.problem.experiment['fluor_diode']
        plt.plot(angles_in, exp_fluor, c='red', lw=2,label = 'Experiment')
        plt.plot(angles_in, model_fluor, c='C0', label='Initial Fit')
        
        self.fitaxes_residual1 = self.fitaxes[1].twinx()
        self.fitaxes_residual1.plot(angles_in,residual_fluor, '.', c = 'grey')
        plt.ylabel('Residual')

        plt.tight_layout()
        
        self.devfig, self.devaxes  = plt.subplots(2,1,figsize=(7, 7))
        self.devfig.suptitle('Parameter Development Plot')
        
    def logging(self,parameters, iteration_counter, residual, problem):
        
        # Log parameters
        self.df_par = self.df_par.append(pd.Series({par: parameters[par].value for par in parameters}, name = iteration_counter))
        
        # Sum residuals
        L = len(problem.angles_in)
        residuals = {'Refl':np.sum(residual[L:]),'Fluor':np.sum(residual[:L])}
        print(residuals)
        s_resid = pd.Series(residuals, name = iteration_counter)
        
        # Log Residuals
        self.df_resid = self.df_resid.append(s_resid)
        
        print(f'Iteration {iteration_counter} complete. Residuals: {residuals}')
        
        # Abort Fit condition
        if iteration_counter > self.maxiter:
            return True
        
        
    def final_plot(self):
        parameters = self.parameters
        ### Add final fits to the plot
        model_fluor = xlf.abs2(self.problem.fluorescence_I_angle_in_dependent)/ parameters['I_fluorescence'].value
        model_refl = np.mean(xlf.abs2(self.problem.reflectivity),0) / parameters['I_reflectivity'].value
        residual_refl = (xlf.norm(self.problem.experiment['refl'] - model_refl ))  * parameters['weight_reflectivity'].value
        residual_fluor = (xlf.norm(self.problem.experiment['fluor_diode'] - model_fluor))  * parameters['weight_fluorescence'].value

        angles_in = xlf.rad2deg(self.problem.angles_in)
        plt.sca(self.fitaxes[0])
        plt.plot(angles_in, model_fluor, c='C1', lw = 2, label='Final Fit')
        self.fitaxes_residual0.plot(angles_in,residual_refl, '.', c = 'k', label = 'Final Residual')
        plt.legend()

        plt.sca(self.fitaxes[1])
        plt.plot(angles_in, model_fluor, c='C1', label='Final Fit')
        self.fitaxes_residual1.plot(angles_in,residual_refl, '.', c = 'k', label = 'Final Residual')
        plt.legend()
        
        
        ### Plot development of fit parameters and residuals
            
        iteration_axes = self.df_resid.index
        ### Parameters
        plt.sca(self.devaxes[0])
        for parname in self.parameters:
            par = self.parameters[parname]
            if par.vary == False:
                continue
                
            def normpar(par, values):
                """
                Normalize a parameter to a range between 0 and 1
                """
                return (values-par.min)/(par.max-par.min)
            
            parlabel = f'{parname} = {par.value:.2e} [{par.min:.2e}  to {par.max:.2e}]'
            plt.plot(iteration_axes, 100*normpar(par, self.df_par[parname]),'.-', label = parlabel)
            
            
        plt.ylim(0,100)    
        plt.legend(fontsize = 7)
        plt.xlabel('No of Iterations')
        plt.ylabel('Parameter value within its range / \%')
        
        ### Residuals
        plt.sca(self.devaxes[1])
        plt.plot(iteration_axes, self.df_resid['Refl'],'.-', label = 'Reflectivity Residual')
        plt.plot(iteration_axes, self.df_resid['Fluor'],'.-', label = 'Fluorescence Residual')
        plt.plot(iteration_axes, self.df_resid['Refl']+self.df_resid['Fluor'],'.-', label = 'Total Residual')
        plt.legend()
        plt.xlabel('No of Iterations')
        plt.ylabel('Residual Value')
