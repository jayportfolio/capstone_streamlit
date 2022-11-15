import math

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from time import time
import json
from datetime import datetime

from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor


def assert_ok(updated, expected__json_filename, testname=""):
    from deepdiff import DeepDiff

    with open(expected__json_filename) as f:
        expected1 = json.loads(f.read())

        answer = DeepDiff(updated, expected1)
        failed = []
        for key00, keys in answer.items():
            if key00 == 'values_changed':
                for key, value in keys.items():
                    if key not in ["root['TEST']['date']", "root['TEST']['first run']"]:
                        failed.append(key)
            else:
                failed.append(key00)
        if failed != []:
            print(testname, ":", failed, 'should be empty')
            print()
            print("updated json")
            print(updated)
        assert failed == []


def test_module():
    if False:
        pass
    elif False:
        trial_df = pd.read_csv('data/final/df_listings_v09.csv')
        feature_engineer(trial_df, version=3)
    elif True:
        using_test_framework = True

        updated = {}
        update_results(updated, make_result(score=10, time=0.1), key='TEST', directory='./offline/')
        if using_test_framework: assert_ok(updated, 'offline/results_test_results/expected1.json', testname='test 1')

        update_results(updated, make_result(score=5, time=0.05), key='TEST', directory='./offline/')
        if using_test_framework: assert_ok(updated, 'offline/results_test_results/expected2.json', testname='test 2')

        update_results(updated, make_result(score=20, time=0.2), key='TEST', directory='./offline/')
        if using_test_framework: assert_ok(updated, 'offline/results_test_results/expected3.json', testname='test 3')

        update_results(updated,  make_result(score=1, time=0.01), key='TEST', directory='./offline/')
        if using_test_framework: assert_ok(updated, 'offline/results_test_results/expected4.json', testname='test 4')

        update_results(updated, make_result(score=20, time=0.2), key='TEST', directory='./offline/')
        if using_test_framework: assert_ok(updated, 'offline/results_test_results/expected5.json', testname='test 5')

        update_results(updated, make_result(score=20, time=0.2, vary='_vary'), key='TEST', directory='./offline/')
        if using_test_framework: assert_ok(updated, 'offline/results_test_results/expected6.json', testname='test 6')

        update_results(updated, make_result(score=20, time=200.0, vary='_vary'), key='TEST', directory='./offline/')
        if using_test_framework: assert_ok(updated, 'offline/results_test_results/expected7.json', testname='test 7')

        update_results(updated, make_result(score=20, time=0.00002, vary='_vary'), key='TEST', directory='./offline/')
        if using_test_framework: assert_ok(updated, 'offline/results_test_results/expected8.json', testname='test 8')

        update_results(updated, make_result(score=20, time=0.02), key='TEST', directory='./offline/')
        if using_test_framework: assert_ok(updated, 'offline/results_test_results/expected9.json', testname='test 9')

        update_results(updated, make_result(score=20, time=0.00002), key='TEST', directory='./offline/')
        if using_test_framework: assert_ok(updated, 'offline/results_test_results/expected10.json', testname='test 10')

    elif False:
        get_hyperparameters('catboost', False)
    else:
        pass


def make_result(score, time, vary=""):
    return {
        # "Mean Absolute Error Accuracy": score,
        # "Mean Squared Error Accuracy": score,
        # "R square Accuracy": score,
        # "Root Mean Squared Error": score,
        '_score': score,
        '_train time': time,
        # "best params": {
        #    "param1": "param1_" + str(score),
        #    "param2": "param2_" + str(score),
        #    "param3": "param3_" + str(score),
        # },
        "date": str(datetime.now()),
        # "first run": "2022-11-06 22:13:02.393884",
        '_params': {
            "param1": "param1_" + str(score) + vary,
            "param2": "param2_" + str(score),
            "param3": "param3_" + str(score),
        },
        "random_state": 101
    }


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


def tidy_dataset(df, version: int) -> pd.DataFrame:
    if version >= 2:
        df = df[df['sharedOwnership'] == False]

    return df


def preprocess(df, version: int) -> pd.DataFrame:
    version_number = int(version)

    if version_number == 2:
        df['location.latitude'] = pd.to_numeric(df['location.latitude'], 'coerce').dropna().astype(float)
        df = df[(df['location.longitude'] <= 10)]
        df = df[(df['bedrooms'] <= 10)]
        df = df[df['bathrooms'] <= 5]
        df = df[(df['nearestStation'] <= 20)]

    elif version_number == 3 or version_number == 4:
        df = df[(df['longitude_deviation'] <= 1)]
        df = df[(df['bedrooms'] <= 10)]
        df = df[df['bathrooms'] <= 5]
        df = df[(df['nearestStation'] <= 20)]

    elif version_number == 5:
        df['location.latitude'] = pd.to_numeric(df['location.latitude'], 'coerce').dropna().astype(float)
        df = df[(df['location.longitude'] <= 10)]

        df = df[(df['longitude_deviation'] <= 1)]
        df = df[(df['bedrooms'] <= 10)]
        df = df[df['bathrooms'] <= 5]
        df = df[(df['nearestStation'] <= 20)]

    elif version_number <= 12:
        df['location.latitude'] = pd.to_numeric(df['location.latitude'], 'coerce').dropna().astype(float)

        df = df[(df['bedrooms'] <= 7)]
        df = df[df['bathrooms'] <= 5]

        df = df[(df['nearestStation'] <= 7.5)]

        df = df[(df['location.longitude'] <= 1)]
        df = df[(df['longitude_deviation'] <= 1)]

    else:
        raise ValueError(f'no columns data available for version {version_number}')

    return df


def feature_engineer(df, version: int) -> pd.DataFrame:
    version_number = int(version)

    if version_number >= 3:
        df['location.latitude'] = pd.to_numeric(df['location.latitude'], 'coerce').dropna().astype(float)
        df['location.longitude'] = pd.to_numeric(df['location.longitude'], 'coerce').dropna().astype(float)

        average_latitude0 = df['location.latitude'].mean()
        average_longitude0 = df['location.longitude'].mean()

        average_latitude1 = df['location.latitude'].median()
        average_longitude1 = df['location.longitude'].median()

        average_latitude2 = 51.4626624
        average_longitude2 = -0.0651048

        df['latitude_deviation'] = abs(df['location.latitude'] - average_latitude1)
        df['longitude_deviation'] = abs(df['location.longitude'] - average_longitude1)

        df['latitude_deviation2'] = abs(df['location.latitude'] - average_latitude2)
        df['longitude_deviation2'] = abs(df['location.longitude'] - average_longitude2)

    if version_number in [9, 10, 11, 12]:
        exploded_features_df = (
            df['reduced_features'].explode()
            .str.get_dummies(',').sum(level=0).add_prefix('feature__')
        )
        df = df.drop('reduced_features', 1).join(exploded_features_df)

    if version_number in [11, 12]:
        dailmail = ['garden', 'central heating', 'parking', 'off road', 'shower', 'cavity wall insulation',
                    'wall insulation', 'insulation', 'insulat', 'dining room', 'garage', 'en-suite', 'en suite']
        common_knowledge = ['penthouse', 'balcony']
        ideal_home = ['double-glazing', 'double glazing', 'off-road parking', 'security', 'patio', 'underfloor heating',
                      'marble']
        discarded = ['signal', 'secure doors', 'secure door', 'outdoor lighting', 'bathtub', 'neighbours', ]

        keywords = []
        keywords.extend(dailmail)
        keywords.extend(common_knowledge)
        keywords.extend(ideal_home)

        import re
        spice_df = pd.DataFrame(dict(('feature__2__' + spice, df.keyFeatures.str.contains(spice, re.IGNORECASE))
                                     for spice in keywords))
        df = df.merge(spice_df, how='outer', left_index=True, right_index=True)

    return df


def create_train_test_data(df_orig, categories, RANDOM_STATE=[], p_train_size=0.9, return_index=False, drop_nulls=True, no_dummies=False):
    df = df_orig.copy()

    if drop_nulls:
        df.dropna(inplace=True)

    if return_index:
        df.reset_index(inplace=True)

    if not no_dummies:
        for column in categories:
            df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
            df.drop([column], axis=1, inplace=True)  # now drop the original column (you don't need it anymore),
    else:
        # one_hot_max_size
        # for column in categories:
        #    df[column] = df[column].astype('category')
        pass

    ins = df.pop('index')
    df.insert(1, 'index2', ins)
    df.insert(0, 'index', ins)

    df_features = df[df.columns[2:]]
    df_labels = df.iloc[:, 0:2]

    features = df[df.columns[2:]].values
    labels = df.iloc[:, 0:2].values

    if not return_index:
        return train_test_split(features, labels, train_size=p_train_size, random_state=RANDOM_STATE)
    else:
        X_train1, X_test1, y_train1, y_test1 = train_test_split(features, labels, train_size=p_train_size,
                                                                random_state=RANDOM_STATE)
        X_train_index = X_train1[:, 0].reshape(-1, 1)
        y_train_index = y_train1[:, 0].reshape(-1, 1)
        X_test_index = X_test1[:, 0].reshape(-1, 1)
        y_test_index = y_test1[:, 0].reshape(-1, 1)
        X_train1 = X_train1[:, 1:]
        y_train1 = y_train1[:, 1].reshape(-1, 1)
        X_test1 = X_test1[:, 1:]
        y_test1 = y_test1[:, 1].reshape(-1, 1)

        return X_train1, X_test1, y_train1, y_test1, X_train_index, X_test_index, y_train_index, y_test_index, df_features, df_labels


def create_train_test_data_XXX(df_orig, categories, RANDOM_STATE=[], p_train_size=0.9, return_index=False, drop_nulls=True):
    df = df_orig.copy()

    if drop_nulls:
        df.dropna(inplace=True)

    if return_index:
        df.reset_index(inplace=True)

    for column in categories:
        df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
        df.drop([column], axis=1, inplace=True)  # now drop the original column (you don't need it anymore),

    ins = df.pop('index')
    df.insert(1, 'index2', ins)
    df.insert(0, 'index', ins)

    features = df[df.columns[2:]].values
    labels = df.iloc[:, 0:2].values

    if not return_index:
        return train_test_split(features, labels, train_size=p_train_size, random_state=RANDOM_STATE)
    else:
        X_train1, X_test1, y_train1, y_test1 = train_test_split(features, labels, train_size=p_train_size,
                                                                random_state=RANDOM_STATE)
        X_train_index = X_train1[:, 0].reshape(-1, 1)
        y_train_index = y_train1[:, 0].reshape(-1, 1)
        X_test_index = X_test1[:, 0].reshape(-1, 1)
        y_test_index = y_test1[:, 0].reshape(-1, 1)
        X_train1 = X_train1[:, 1:]
        y_train1 = y_train1[:, 1].reshape(-1, 1)
        X_test1 = X_test1[:, 1:]
        y_test1 = y_test1[:, 1].reshape(-1, 1)

        return X_train1, X_test1, y_train1, y_test1, X_train_index, X_test_index, y_train_index, y_test_index


def get_chosen_model(key):
    models = {
        "XG Boost".lower(): XGBRegressor(seed=20),
        "Linear Regression (Ridge)".lower(): Ridge(),
        "knn": KNeighborsRegressor(),
        "decision tree": DecisionTreeRegressor(),
        "random forest": RandomForestRegressor(),
        "CatBoost".lower(): CatBoostRegressor(objective='RMSE'),
        # "CatBoost".lower(): CatBoostRegressor(objective='R2'),
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

    elif key.lower() in ['catboost', 'random forest', "Linear Regression (Ridge)".lower()]:

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


def get_best_estimator_average_time(best_estimator_pipe, X_train, y_train, debug=False):
    timings = []

    max_time_iter = 5 if debug else 1

    for i in range(0, max_time_iter):
        t0 = time()
        best_estimator_pipe.fit(X_train, y_train)
        timings.append(time() - t0)
        if time() - t0 > 30: print(i, ":", time() - t0)

    print(timings)
    average_time = sum(timings) / len(timings)

    return average_time


def get_results():
    results_filename = '../../../results/results.json'

    with open(results_filename) as f:
        raw_audit = f.read()
    results_json = json.loads(raw_audit)
    return results_json


def update_results(saved_results_json, new_results, key, directory='../../../results/', aim='maximize'):
    bad = []
    for each in ['_params', '_score', '_train time', 'date', 'random_state']:
        if each not in list(new_results.keys()):
            bad.append(each)
        if len(bad) > 0:
            raise ValueError(str(bad) + ' should be in the results array')

    first_run_date = str(datetime.now())
    if saved_results_json is not None and key in saved_results_json:
        old_results = saved_results_json[key]

    try:
        first_run_date = old_results['date']
        first_run_date = old_results['first run']
    except:
        pass

    max_score = -1000
    try:
        max_score = max(max_score, old_results['_score'])
        max_score = max(max_score, old_results['best score'])
    except:
        pass

    new_results['first run'] = first_run_date

    if key not in saved_results_json:
        put_new_in_best(new_results)
        this_model_is_best = True
    elif max_score > new_results['_score']:
        put_old_best_in_best(new_results, old_results)

        this_model_is_best = False
    elif max_score == new_results['_score']:

        if old_results['best params'] == new_results['_params'] and new_results['_train time'] <= old_results['best time']:

            put_new_in_best(new_results)

            this_model_is_best = True

        elif old_results['best params'] != new_results['_params'] and new_results['_train time'] <= old_results['best time']:

            put_new_in_best(new_results)
            new_results['best is shared'] = True

            this_model_is_best = True

        elif old_results['best params'] == new_results['_params'] or old_results['best time'] > new_results['_train time'] * 3:
            put_old_best_in_best(new_results, old_results) ## was best2

            this_model_is_best = False

        else:
            put_old_best_in_best(new_results, old_results) ## was best2
            new_results['best is shared'] = True

            this_model_is_best = False

    else:
        put_new_in_best(new_results)

        this_model_is_best = True

    saved_results_json[key] = new_results.copy()

    results_filename = directory + 'results.json'
    with open(results_filename, 'w') as file:
        file.write(json.dumps(saved_results_json, indent=4, sort_keys=True))

    return this_model_is_best


def put_old_best_in_best2XXX(new_results, old_results):
    new_results['best params'] = old_results['best params']
    new_results['best score'] = old_results['best score']
    new_results['best time'] = old_results['_train time']
    new_results['suboptimal'] = 'pending'


def put_old_best_in_best(new_results, old_results):
    new_results['best params'] = old_results['best params']
    new_results['best score'] = old_results['best score']
    new_results['best time'] = old_results['_train time']
    new_results['suboptimal'] = 'suboptimal'


def put_new_in_best(new_results):
    new_results['best params'] = new_results['_params']
    new_results['best score'] = new_results['_score']
    new_results['best time'] = new_results['_train time']
    new_results['suboptimal'] = 'pending'


if __name__ == '__main__':
    test_module()
