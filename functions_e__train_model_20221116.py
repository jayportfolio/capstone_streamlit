from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


from time import time

def make_modelling_pipeline(model, DATA_DETAIL):
    if 'no scale' in DATA_DETAIL:
        pipe = Pipeline([
            ('model', model)
        ])
    else:
        pipe = Pipeline([
            # ('mms', MinMaxScaler()),
            ('std_scaler', StandardScaler()),
            ('model', model)
        ])
    return pipe


def build_model(algorithm, drop_nulls=False):
    X_train, X_test, y_train, y_test = this_test_data(drop_nulls=drop_nulls)

    if algorithm == 'Decision Tree':
        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)

    elif algorithm == 'Linear Regression':
        model = LinearRegression()
        model.fit(X_train, y_train)

    elif algorithm == 'HistGradientBoostingRegressor':
        model = HistGradientBoostingRegressor()
        model.fit(X_train, y_train)

    elif algorithm == 'Deep Neural Network':

        import tensorflow as tf

        from tensorflow import keras
        from tensorflow.keras import layers

        print(tf.__version__)

        def build_and_compile_model(norm):
            model = keras.Sequential([
                norm,
                layers.Dense(132, activation='relu'),
                layers.Dense(132, activation='relu'),
                #layers.Dense(400, activation='relu'),
                #layers.Dense(400, activation='relu'),
                #layers.Dense(400, activation='relu'),
                layers.Dense(1)
            ])

            model.compile(loss='mean_absolute_error',
                          optimizer=tf.keras.optimizers.Adam(0.001))
            return model

        normalizer = tf.keras.layers.Normalization(axis=-1)

        dnn_model = build_and_compile_model(normalizer)
        # print(dnn_model.summary())

        # % % time
        history = dnn_model.fit(
            X_train,  # train_features,
            y_train,  # train_labels,
            validation_split=0.2,
            verbose=0, epochs=100)

        # def plot_loss(history):
        #     import matplotlib.pyplot as plt
        #     plt.plot(history.history['loss'], label='loss')
        #     plt.plot(history.history['val_loss'], label='val_loss')
        #     plt.ylim([0, 10])
        #     plt.xlabel('Epoch')
        #     plt.ylabel('Error [MPG]')
        #     plt.legend()
        #     plt.grid(True)
        #
        # plot_loss(history)

        # test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)
        print(dnn_model.evaluate(X_test, y_test, verbose=0))

        model = dnn_model

    elif algorithm == 'Linear Regression (Keras)':
        import tensorflow as tf

        from tensorflow import keras
        from tensorflow.keras import layers

        print(tf.__version__)

        normalizer = tf.keras.layers.Normalization(axis=-1)
        linear_model = tf.keras.Sequential([
            normalizer,
            layers.Dense(units=1)
        ])

        linear_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
            loss='mean_absolute_error')

        # %%time
        history = linear_model.fit(
            X_train,  # train_features,
            y_train,  # train_labels,
            epochs=100,
            # Suppress logging.
            verbose=0,
            # Calculate validation results on 20% of the training data.
            validation_split=0.2)

        # def plot_loss(history):
        #     import matplotlib.pyplot as plt
        #     plt.plot(history.history['loss'], label='loss')
        #     plt.plot(history.history['val_loss'], label='val_loss')
        #     plt.ylim([0, 10])
        #     plt.xlabel('Epoch')
        #     plt.ylabel('Error [MPG]')
        #     plt.legend()
        #     plt.grid(True)
        #
        # plot_loss(history)

        model = linear_model

    elif algorithm == 'Linear Regression (Keras)':
        from tensorflow_estimator.python.estimator.canned.linear import LinearRegressor

        model = LinearRegressor()
        model.fit(X_train, y_train)
    else:
        raise ValueError(algorithm)

    return model

def get_chosen_model(key):
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    from lightgbm import DaskLGBMRegressor

    models = {
        "XG Boost".lower(): XGBRegressor(seed=20),
        "Linear Regression (Ridge)".lower(): Ridge(),
        "knn": KNeighborsRegressor(),
        "decision tree": DecisionTreeRegressor(),
        "random forest": RandomForestRegressor(),
        "CatBoost".lower(): CatBoostRegressor(objective='RMSE'),
        # "CatBoost".lower(): CatBoostRegressor(objective='R2'),
        "Light Gradient Boosting".lower(): LGBMRegressor()
    }
    try:
        return models.get(key.lower())
    except:
        raise ValueError(f'no model found for key: {key}')


def get_hyperparameters(key, use_gpu, prefix='./'):
    if key.lower() == "XG Boost".lower():
        with open(prefix + f'process/z_envs/hyperparameters/{key.lower()}.json') as f:
            hyperparameters = json.loads(f.read())

        if use_gpu:
            # hyperparameters['tree_method'].append('gpuhist')
            ###hyperparameters['n_estimators'].extend([250, 300, 500, 750, 1000])
            ###hyperparameters['gamma'].extend([1000000, 10000000])
            # hyperparameters['learning_rate'].extend([0.3, 0.01, 0.1, 0.2, 0.3, 0.4])
            # hyperparameters['early_stopping_rounds'].extend([1, 5, 10, 100])
            pass

    elif key.lower() in ['catboost', 'random forest', "Linear Regression (Ridge)".lower(), "Light Gradient Boosting".lower()]:

        with open(prefix + f'process/z_envs/hyperparameters/{key.lower()}.json') as f:
            hyperparameters = json.loads(f.read())

    elif key.lower() == 'knn':

        hyperparameters = {
            # 'objective': 'reg:squarederror',
            # 'max_depth': [1, 3, 6, 10, 30],
            # 'n_estimators': 100,
            # 'tree_method': ['auto', 'approx', 'hist', 'exact'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [3.30, 300, 3000],
            'metric': ['minkowski', 'precomputed'],
            # 'metric_params' : 'abc',
            'n_jobs': [-1, 1, 2],
            'n_neighbors': [3, 4, 5, 7, 9, 13, 21, 35],
            'p': [1, 2],
            'weights': ['uniform', 'distance']
            # 'verbosity': 1
        }
    elif key.lower() == 'decision tree':

        hyperparameters = {
            'splitter': ['best', 'random'],
            'random_state': None,
            'min_weight_fraction_leaf': [0.0, 0.1, 0.25, 0.5],  # , 1, 5],
            'min_samples_split': [2, 4, 8, 50, 100, 200, 500],  # , .5, 1]
            'min_samples_leaf': [1, 0.25, 0.5, 1.5, 2, 4, 8, 50],
            'min_impurity_decrease': [0.0, 0.1, 0.25, 1, 5],
            'max_leaf_nodes': [None, 2, 5, 10, 50, 100, 200, 500],  # 1]
            'max_features': [None, 1.0, 'sqrt', 'log2', .5, .25, .1, 2],
            'max_depth': [None, 1, 2, 5, 10, 50],
            # 'criterion': ['gini','entropy','log_loss'], gini and entropy apply to classifier, not regressor
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],  # ,'log_loss'],
            'ccp_alpha': [0.0, 0.05, 0.1, 0.25, 1, 5],  # Cost Complexity Pruning, ref 13.3.1

        }

    else:
        raise ValueError("couldn't find hyperparameters for:", key)

    return hyperparameters


def get_cv_params(options_block, debug_mode=True, override_cv=None, override_niter=None, override_verbose=None, override_njobs=None):
    global param_options, cv, n_jobs, verbose, refit, iter
    param_options = {}
    for each in options_block:
        if type(options_block[each]) == list:
            param_options['model__' + each] = options_block[each]
        elif options_block[each] == None:
            # print (f'skipping {each} because value is {options_block[each]}')
            param_options['model__' + each] = [options_block[each]]
        else:
            param_options['model__' + each] = [options_block[each]]

    # cv = 3
    cv = override_cv if override_cv else 3
    # n_iter = 100
    n_iter = override_niter if override_niter else 100
    # verbose = 3 if debug_mode else 1
    verbose = override_verbose if override_verbose else 3 if debug_mode else 1
    refit = True
    # n_jobs = 1
    n_jobs = override_njobs if override_njobs else 1 if debug_mode else 3

    return param_options, cv, n_jobs, refit, n_iter, verbose


def fit_model_with_cross_validation(gs, X_train, y_train, fits):
    pipe_start = time()
    cv_result = gs[0].fit(X_train, y_train)
    pipe_end = time()
    average_time = round((pipe_end - pipe_start) / (fits), 2)
    # xxx print(f"{average_time} seconds per fit") # not correct if declared fits was overstated
    # print(f"average fit time = {cv_result.cv_results_.mean_fit_time}s")
    # print(f"max fit time = {cv_result.cv_results_.mean_fit_time.max()}s")
    # print(f"average fit/score time = {round(cv_result.cv_results_.mean_fit_time,2)}s/{round(cv_result.cv_results_.mean_score_time,2)}s")

    print(f"Total fit/CV time      : {int(pipe_end - pipe_start)} seconds   ({pipe_start} ==> {pipe_end})")
    print()
    print(f'average fit/score time = {round(cv_result.cv_results_["mean_fit_time"].mean(), 2)}s/{round(cv_result.cv_results_["mean_score_time"].mean(), 2)}s')
    print(f'max fit/score time     = {round(cv_result.cv_results_["mean_fit_time"].max(), 2)}s/{round(cv_result.cv_results_["mean_score_time"].max(), 2)}s')
    print(f'refit time             = {round(cv_result.refit_time_, 2)}s')

    return cv_result, average_time, cv_result.refit_time_, len(cv_result.cv_results_["mean_fit_time"])

def automl_step(param_options, vary):
    for key, value in param_options.items():
        #print(key, value, vary)
        if key != vary and key != 'model__' + vary:
            param_options[key] = [param_options[key][0]]
    return param_options


if False:
    param_options = automl_step(param_options, vary='gamma')
    print(f'cv={cv}, n_jobs={n_jobs}, refit={refit}, n_iter={n_iter}, verbose={verbose}')