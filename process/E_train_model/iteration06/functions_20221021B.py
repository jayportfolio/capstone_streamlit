import math

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def test_module():
    # trial_df = pd.read_csv('../../data/final/df_listings_v02.csv')
    # feature_engineer(trial_df, version=3)
    pass

# csv_directory = "final_split"
# csv_directory = "quick_split"
csv_directory = "DDD"

FINAL_BASIC_FILE = "data/DDD/listings_data_basic_XXX.csv"
FINAL_ENRICHED_FILE = "data/DDD/listings_data_enriched_XXX.csv"
FINAL_JSON_MODEL_FILE = "data/DDD/listings_data_jsonmodel_XXX.csv"
FINAL_JSON_META_FILE = "data/DDD/listings_data_jsonmeta_XXX.csv"
FINAL_RECENT_FILE = "data/source/df_listings.csv"
FINAL_RECENT_FILE_SAMPLE = "data/sample/df_listings_sample.csv"


def set_csv_directory(update_csv_directory):
    global csv_directory
    csv_directory = update_csv_directory


def get_source_dataframe(IN_COLAB, VERSION, rows=0, folder_prefix='../../../'):
    retrieval_type = None

    filename = f'df_listings_v{VERSION}.csv'
    remote_pathname = f'https://raw.githubusercontent.com/jayportfolio/capstone_streamlit/main/data/final/{filename}'
    df_pathname_raw = folder_prefix + f'data/source/{filename}'
    df_pathname_tidy = folder_prefix + f'data/final/{filename}'

    if IN_COLAB:
        inDF = pd.read_csv(remote_pathname, on_bad_lines='error', index_col=0)
        retrieval_type = 'tidy'
        print('loaded data from', folder_prefix + remote_pathname)
    else:
        inDF = pd.read_csv(df_pathname_tidy, on_bad_lines='error', index_col=0)
        retrieval_type = 'tidy'
        print('loaded data from', df_pathname_tidy)

    if rows and rows > 0:
        inDF = inDF[:rows]
    return inDF, retrieval_type



def this_test_data(VERSION,  test_data_only=False, drop_nulls=True):
    suffix = "_no_nulls" if drop_nulls else ""

    try:
        if not test_data_only:
            X_train = np.loadtxt(f"train_test/X_train{suffix}.csv", delimiter=",")
            y_train = np.loadtxt(f"train_test/y_train{suffix}.csv", delimiter=",")

        X_test = np.loadtxt(f"train_test/X_test{suffix}.csv", delimiter=",")
        y_test = np.loadtxt(f"train_test/y_test{suffix}.csv", delimiter=",")
    except:
        df, retrieval_type = get_source_dataframe(IN_COLAB=False, VERSION=VERSION, folder_prefix='')

        if drop_nulls:
            df.dropna(inplace=True)

        # xxxfeatures = df[df.columns[:-1]].values
        # features = df[FEATURES].values
        # labels = df[LABEL].values
        # X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.9, random_state=RANDOM_STATE)
        X_train, X_test, y_train, y_test =  tt_split (VERSION, df)

        print('test_data_only', test_data_only)
        print('drop_nulls', drop_nulls)

        if not test_data_only:
            suffix = '_no_nulls' if drop_nulls else ''
            print('suffix:', suffix)
            print('text:', f"train_test/X_train{suffix}.csv")
            print()
            print(X_train)
            print()
            np.savetxt("train_test/X_train_no_nulls.csv", X_train, delimiter=",")
            np.savetxt(f"train_test/y_train{suffix}.csv", y_train, delimiter=",")

        #np.savetxt(f"train_test/X_test{suffix}.csv", X_test[:20], delimiter=",")
        np.savetxt(f"train_test/X_test{suffix}.csv", X_test, delimiter=",")
        #np.savetxt(f"train_test/y_test{suffix}.csv", y_test[:20], delimiter=",")
        np.savetxt(f"train_test/y_test{suffix}.csv", y_test, delimiter=",")

    if not test_data_only:
        return X_train, X_test, y_train, y_test

    return X_test, y_test

def tt_split(VERSION, df, RANDOM_STATE=101, LABEL='Price'):

    columns, booleans, floats, categories = get_columns(version=VERSION)

    for column in categories:
        df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
        df.drop([column], axis=1, inplace=True)  # now drop the original column (you don't need it anymore),

    # features = df[df.columns[:-1]].values
    features = df[df.columns[1:]].values
    # features = df[FEATURES].values
    labels = df[LABEL].values
    X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.9, random_state=RANDOM_STATE)
    return X_train, X_test, y_train, y_test


def get_df(file, folder_prefix=''):
    df_array = []
    for n in range(10):
        prefix = '00' if n <= 10 else '0'

        filename = folder_prefix + file.replace('XXX', prefix + str(n)).replace('DDD', csv_directory)

        from os.path import exists
        file_exists = exists(filename)

        if file_exists:
            if n == 0:
                merged_df = pd.read_csv(filename, on_bad_lines='skip')
            else:
                # each_df = pd.read_csv(filename, on_bad_lines='skip')
                # df_array.append(each_df)
                merged_df = pd.concat([merged_df, pd.read_csv(filename, on_bad_lines='skip')])
                # print(n)

            # print(f'error on {file} at split {prefix}')
        else:
            if n == 1:
                raise LookupError("didn't find ANY matching files! filename: " + filename)
            # print(f'{n + 1} splits for {file}')
            break

    return merged_df


def add_supplements(property_dataset):
    property_dataset['Price'] = pd.to_numeric(property_dataset['Price'], 'coerce').dropna().astype(int)

    # do any necessary renames, and some preliminary feature engineering
    try:
        property_dataset = property_dataset.rename(index=str, columns={"Station_Prox": "distance_to_any_train"})
    except:
        pass

    try:
        property_dataset['borough'] = property_dataset["borough"].str.extract("\('(.+)',")
    except:
        pass

    def simplify(array_string):
        try:
            array = array_string.split("/")  # a list of strings
            return array[0]
        except:
            pass

    try:
        property_dataset['propertyType'] = property_dataset['analyticsProperty.propertyType'].apply(simplify)
    except:
        pass

    try:
        property_dataset['coarse_compass_direction'] = property_dataset["address.outcode"].str.extract("([a-zA-Z]+)")
    except:
        pass

    try:
        property_dataset['sq_ft'] = property_dataset["size"].str.extract("(\d*) sq. ft.")
    except:
        pass

    property_dataset = property_dataset[(property_dataset['Price'] >= 100000) & (property_dataset['Price'] <= 600000)]

    property_dataset['sharedOwnership'] = (
            (property_dataset['sharedOwnership.sharedOwnership'] == True) |
            (property_dataset['analyticsProperty.priceQualifier'] == 'Shared ownership') |
            (property_dataset['keyFeatures'].str.contains('shared ownership'))
    )

    property_dataset['keyFeatures'] = property_dataset['keyFeatures'].str.lower()
    property_dataset['text.description'] = property_dataset['text.description'].str.lower()

    property_dataset['sharedOwnership'] = (property_dataset['sharedOwnership'])
    property_dataset['sharedOwnership'] = (property_dataset['sharedOwnership']) | (
            property_dataset['sharedOwnership.sharedOwnership'] == 1)
    property_dataset['sharedOwnership'] = (property_dataset['sharedOwnership']) | (
            property_dataset['analyticsProperty.priceQualifier'] == 'Shared ownership')

    property_dataset['sharedOwnership'] = (property_dataset['sharedOwnership']) | (
        property_dataset['keyFeatures'].str.contains('shared ownership'))
    property_dataset['sharedOwnership'] = (property_dataset['sharedOwnership']) | (
            (property_dataset['keyFeatures'].str.contains('share')) & (
        property_dataset['keyFeatures'].str.contains('%')))

    property_dataset['sharedOwnership'] = (property_dataset['sharedOwnership']) | (
        property_dataset['text.description'].str.contains('shared ownership'))
    property_dataset['sharedOwnership'] = (property_dataset['sharedOwnership']) | (
            (property_dataset['text.description'].str.contains('share')) & (
        property_dataset['text.description'].str.contains('%')))

    # property_dataset['sharedOwnership'] = (property_dataset['sharedOwnership']) | (property_dataset['sharedownership_in_description'])

    def share_percentage(df_row):
        #        print(df_row)
        if df_row['sharedOwnership.sharedOwnership']:
            if type(df_row['sharedOwnership.ownershipPercentage']) in [int, float] and not math.isnan(
                    df_row['sharedOwnership.ownershipPercentage']):
                return df_row['sharedOwnership.ownershipPercentage']
            else:
                return None
        else:
            return 100

    try:
        property_dataset['sharePercentage'] = property_dataset.apply(share_percentage, axis=1)
    except:
        pass

    def stations(station_list_string, requested_type):
        # print('stations')
        # pass
        # station_list = json.loads(station_list_string)
        import ast
        station_list = ast.literal_eval(station_list_string)

        # print('---')
        # print(station_list)

        # NATIONAL_TRAIN
        # LIGHT_RAILWAY
        # TRAM
        for station in station_list:

            if station['types'] not in [
                ['NATIONAL_TRAIN'],
                ['LONDON_UNDERGROUND'],
                ['LIGHT_RAILWAY'],
                ['LONDON_OVERGROUND'],
                ['TRAM'],
                ['CABLE_CAR'],
                ['LONDON_UNDERGROUND', 'LIGHT_RAILWAY'],
                ['LIGHT_RAILWAY', 'LONDON_OVERGROUND'],
                ['LONDON_UNDERGROUND', 'LONDON_OVERGROUND'],
                ['NATIONAL_TRAIN', 'LONDON_UNDERGROUND', 'LONDON_OVERGROUND'],
                ['NATIONAL_TRAIN', 'LONDON_UNDERGROUND'],
                ['NATIONAL_TRAIN', 'LIGHT_RAILWAY'],
                ['NATIONAL_TRAIN', 'TRAM'],
                ['NATIONAL_TRAIN', 'LONDON_OVERGROUND'],
                ['NATIONAL_TRAIN', 'TRAM', 'LONDON_OVERGROUND'],
                ['NATIONAL_TRAIN', 'LONDON_UNDERGROUND', 'LIGHT_RAILWAY'],
                ['NATIONAL_TRAIN', 'LONDON_UNDERGROUND', 'TRAM'],
                ['NATIONAL_TRAIN', 'LONDON_UNDERGROUND', 'LIGHT_RAILWAY', 'LONDON_OVERGROUND'],
            ]:
                print(f"WARNING: Station type not found: {station['types']}: {station}")

            if requested_type == 'any':
                # print(station)
                return station['distance']
            elif requested_type in station['types']:
                return station['distance']
            elif requested_type == "overground" and (
                    'NATIONAL_TRAIN' in station['types'] or 'LONDON_OVERGROUND' in station[
                'types'] or 'LIGHT_RAILWAY' in station['types']):
                return station['distance']
            elif requested_type == "underground combined" and 'LONDON_UNDERGROUND' in station['types'] and len(
                    station['types']) > 1:
                return station['distance']
            else:
                pass

        return 99

    try:
        #        property_dataset['sharePercentage'] = property_dataset.apply(share_percentage, axis=1)
        # property_dataset['nearestUnderground'] = property_dataset['nearestStations'].apply(stations, args='underground')
        property_dataset['nearestStation'] = property_dataset['nearestStations'].apply(stations, args=['any'])
        property_dataset['nearestTram'] = property_dataset['nearestStations'].apply(stations, args=['TRAM'])
        property_dataset['nearestUnderground'] = property_dataset['nearestStations'].apply(stations,
                                                                                           args=['LONDON_UNDERGROUND'])
        property_dataset['nearestOverground'] = property_dataset['nearestStations'].apply(stations,
                                                                                          args=['overground'])

    # sample_df['new_id'] = sample_df[id_label].apply(convert_id_to_hash2, args=[dictionary])

    except:
        pass

    return property_dataset


def tidy_dataset(df, version: int) -> pd.DataFrame:
    if version >= 2:
        df = df[df['sharedOwnership'] == False]

    return df


def get_columns(version: int) -> pd.DataFrame:
    version_number = int(version)

    if version_number == 2:
        booleans = []
        floats = ['bedrooms', 'bathrooms', 'nearestStation', 'location.latitude', 'location.longitude']
        categories = ['tenure.tenureType']

    elif version_number == 3 or version_number == 4:
        booleans = []
        floats = ['bedrooms', 'bathrooms', 'nearestStation', 'latitude_deviation', 'longitude_deviation']
        categories = ['tenure.tenureType']

    elif version_number <= 6:
        booleans = []
        floats = ['bedrooms', 'bathrooms', 'nearestStation', 'location.latitude', 'location.longitude',
                  'latitude_deviation', 'longitude_deviation']
        categories = ['tenure.tenureType']

    else:
        raise ValueError(f'no columns data available for version {version}')

    columns = []
    columns.extend(booleans)
    columns.extend(floats)
    columns.extend(categories)

    return (columns, booleans, floats, categories)


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

    elif version_number == 6:
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

        # average_latitude = df['location.latitude'].mean()
        # average_longitude = df['location.longitude'].mean()
        # print(average_latitude)
        # print(average_longitude)
        # print()
        average_latitude1 = df['location.latitude'].median()
        average_longitude1 = df['location.longitude'].median()
        # print(average_latitude1)
        # print(average_longitude1)
        # print()
        # 51.499672
        # -0.10444
        average_latitude2 = 51.4626624
        average_longitude2 = -0.0651048

        # average_latitude = (df['location.latitude'].max() + df['location.latitude'].min())/2
        # average_longitude = (df['location.longitude'].max() + df['location.longitude'].min())/2
        # print(average_latitude)
        # print(average_longitude)
        # print()

        df['latitude_deviation'] = abs(df['location.latitude'] - average_latitude1)
        df['longitude_deviation'] = abs(df['location.longitude'] - average_longitude1)

        df['latitude_deviation2'] = abs(df['location.latitude'] - average_latitude2)
        df['longitude_deviation2'] = abs(df['location.longitude'] - average_longitude2)

        return df

if __name__ == '__main__':
    test_module()


def get_combined_dataset(HOW, early_duplicates, row_limit=None, verbose=False, folder_prefix=''):
    df_list = get_df(FINAL_BASIC_FILE, folder_prefix)
    # df_indiv = get_df(FINAL_ENRICHED_FILE, testing)
    df_indiv = get_df(FINAL_ENRICHED_FILE, folder_prefix)
    df_meta = get_df(FINAL_JSON_META_FILE, folder_prefix)
    # df_json1 = get_df(LISTING_JSON_MODEL_FILE, on_bad_lines='warn')  # EDIT 29-06-2022: There are bid listings and regular listings. I scrape them seporately and join them here.
    # df_json = get_df(LISTING_JSON_MODEL_FILE)
    df_json = get_df(FINAL_JSON_MODEL_FILE, folder_prefix)

    df_meta['id_copy'] = df_meta['id_copy'].astype(int)
    df_json['id'] = df_json['id'].astype(int)
    df_list.set_index(['ids'], inplace=True)
    df_indiv.set_index(['ids'], inplace=True)
    df_meta.set_index(['id_copy'], inplace=True)
    df_json.set_index(['id'], inplace=True)
    # df_age.set_index(['ids'], inplace=True)

    if HOW == 'no_indexes':
        df_original = df_list \
            .merge(df_json, left_on='ids', right_on='id', how=HOW, suffixes=('', '_model')) \
            .merge(df_meta, left_on='ids', right_on='id_copy', how=HOW, suffixes=('', '_meta'))
        # .merge(df_indiv, on='ids', how='inner', suffixes=('', '_listing')) \
    elif HOW == 'listings_only':
        df_original = df_list
    elif HOW == 'left':  # https://www.statology.org/pandas-merge-on-index/
        df_original = df_list \
            .join(df_json, how=HOW, lsuffix='', rsuffix='_model') \
            .join(df_meta, how=HOW, lsuffix='', rsuffix='_meta')  # \
        # .join(df_indiv, on='ids', how='inner', lsuffix='', rsuffix='_listing') \
        # .join(df_age, how=HOW, lsuffix='', rsuffix='_age')
    elif HOW == 'inner2':  # https://www.statology.org/pandas-merge-on-index/
        df_original = df_list \
            .join(df_json, how='inner', lsuffix='', rsuffix='_model') \
            .join(df_meta, how='inner', lsuffix='', rsuffix='_meta') \
            .join(df_indiv, how='inner', lsuffix='', rsuffix='_listing')
        # .join(df_age, how=HOW, lsuffix='', rsuffix='_age')
    elif HOW == 'inner':  # https://www.statology.org/pandas-merge-on-index/
        df_original = pd.merge(
            pd.merge(pd.merge(df_list, df_indiv, left_index=True, right_index=True, suffixes=('', '_listing'))
                     , df_json, left_index=True, right_index=True, suffixes=('', '_model'))
            , df_meta, left_index=True, right_index=True, suffixes=('', '_meta'))
    else:
        raise LookupError(f"no HOW parameter called {HOW}")

    # df_original.to_csv(vv.QUICK_COMBINED_FILE, mode='w', encoding="utf-8", index=True, header=True)
    del df_list
    # del df_indiv
    del df_json
    del df_meta

    df_original.iloc[:20].to_csv(folder_prefix + 'data/source/df_source_full_sample.csv')
    df_original.to_csv(folder_prefix + 'data/source/df_source_full.csv')

    if row_limit and row_limit > 0:
        # return df_original[:row_limit]
        return df_original.sample(n=row_limit)

    if early_duplicates:
        df_original = df_original[~df_original.index.duplicated(keep='last')]

    return df_original

from datetime import datetime

def update_results(key, saved_results_json, new_results):
    try:
        first_run_date = str(datetime.now())
        first_run_date = saved_results_json[key]['date']
        first_run_date = saved_results_json[key]['first run']
    except:
        pass

    try:
        best_score = -1000
        best_params = 'NOT APPLICABLE'
        best_time = 99999999
        best_score = saved_results_json[key]['Score']
        best_params = saved_results_json[key]['params']
        best_time = saved_results_json[key]['Training Time']
        best_score = saved_results_json[key]['best score']
        best_params = saved_results_json[key]['best params']
        best_time = saved_results_json[key]['best time']
    except:
        pass

    new_results['first run'] = first_run_date

    if key not in saved_results_json:
        new_results['best params'] = new_results['params']
        new_results['best score'] = new_results['Score']
        new_results['best time'] = new_results['Training Time']
        new_results['suboptimal'] = 'pending'

    elif best_score > saved_results_json[key]['Score']:
        new_results['suboptimal'] = 'suboptimal'

    elif best_score == saved_results_json[key]['Score']:
        if saved_results_json[key]['params'] != new_results['params']:
            new_results['best params'] = 'MULTIPLE PARAM OPTIONS'
            new_results['best is shared'] = True
            if new_results['Training Time'] < best_time:
                new_results['best params'] = new_results['params']
                new_results['best score'] = new_results['Score']
                new_results['best time'] = new_results['Training Time']
                new_results['suboptimal'] = 'pending'
            else:
                new_results['best params'] = saved_results_json[key]['params']
                new_results['best score'] = saved_results_json[key]['Score']
                new_results['best time'] = saved_results_json[key]['Training Time']
                new_results['suboptimal'] = 'pending'

        else:
            new_results['best params'] = saved_results_json[key]['params']
            new_results['best score'] = saved_results_json[key]['Score']
            new_results['suboptimal'] = 'pending'

    else:
        new_results['best params'] = saved_results_json[key]['params']
        new_results['best score'] = saved_results_json[key]['Score']
        new_results['suboptimal'] = 'pending'

    saved_results_json[key] = new_results

    return saved_results_json
