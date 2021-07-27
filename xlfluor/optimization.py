import lmfit


def cost_function(cavity_parameters, ):

    problem.update_parameters(cavity_parameters)
    solution = problem.solve()

    model_fluor = solution.fluorescence_curve(cavity_parameters)
    model_refl = solution.fluorescence_curve

    residual = 0.5 * (problem.experiment.refl - model_refl)**2 + 0.5 * (problem.experiment.fluor- model_fluor)**2
    return residual


my_problem = Problem(..., exp_refl, exp_fluor)

minimizer = lmfit.Minimizer(cost_function, params = initial_parameters, fcn_args = (my_problem))

result = minimizer.minimize()


