def automl_step(param_options, vary):
    for key, value in param_options.items():
        #print(key, value, vary)
        if key != vary and key != 'model__' + vary:
            param_options[key] = [param_options[key][0]]
    return param_options


if False:
    param_options = automl_step(param_options, vary='gamma')
    print(f'cv={cv}, n_jobs={n_jobs}, refit={refit}, n_iter={n_iter}, verbose={verbose}')