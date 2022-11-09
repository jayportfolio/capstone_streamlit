import math

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from time import time
import json
from datetime import datetime


def test_module():
    if True:
        pass
    elif False:
        trial_df = pd.read_csv('data/final/df_listings_v09.csv')
        feature_engineer(trial_df, version=3)


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


def create_train_test_data(df_orig, categories, RANDOM_STATE=[], p_train_size=0.9, return_index=False, drop_nulls=True):
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
    print(f"Total fit/CV time: {int(pipe_end - pipe_start)} seconds   ({pipe_start} ==> {pipe_end})")
    average_time = round((pipe_end - pipe_start) / (fits), 2)
    print(f"{average_time} seconds per fit")

    return cv_result, average_time


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


def update_results(saved_results_json, new_results, key):
    first_run_date = str(datetime.now())
    try:
        first_run_date = saved_results_json[key]['date']
        first_run_date = saved_results_json[key]['first run']
    except:
        pass

    max_score = -1000
    try:
        max_score = saved_results_json[key]['Score']
        max_score = saved_results_json[key]['best score']
    except:
        pass

    new_results['first run'] = first_run_date

    if key not in saved_results_json:
        new_results['best params'] = new_results['params']
        new_results['best score'] = new_results['Score']
        new_results['suboptimal'] = 'pending'
    elif max_score > saved_results_json[key]['Score']:
        new_results['suboptimal'] = 'suboptimal'
    elif max_score == saved_results_json[key]['Score']:
        if saved_results_json[key]['params'] != new_results['params']:
            new_results['best params'] = 'MULTIPLE PARAM OPTIONS'
        else:
            new_results['best params'] = saved_results_json[key]['params']
            new_results['best score'] = saved_results_json[key]['Score']
            new_results['suboptimal'] = 'pending'
    else:
        new_results['best params'] = saved_results_json[key]['params']
        new_results['best score'] = saved_results_json[key]['Score']
        new_results['suboptimal'] = 'pending'

    saved_results_json[key] = new_results

    results_filename = '../../../results/results.json'
    with open(results_filename, 'w') as file:
        file.write(json.dumps(saved_results_json, indent=4, sort_keys=True))


if __name__ == '__main__':
    test_module()
