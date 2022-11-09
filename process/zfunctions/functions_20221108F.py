import math

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ast


def test_module():
    if True:
        pass
    elif False:
        trial_df = pd.read_csv('data/final/df_listings_v09.csv')
        feature_engineer(trial_df, version=3)
    else:
        trial_df = pd.read_csv('data/source/df_listings_v09.csv')
        # add_supplements(trial_df)
        dff, aggregate_features = aggregate_keyFeatures(trial_df['keyFeatures'])
        trial_df['reduced_features'] = trial_df['keyFeatures'].apply(reduce_keyFeatures,
                                                                                     args=[dff.index,
                                                                                           get_exploding_features(
                                                                                               version_number=9)])
        trial_df = feature_engineer(trial_df, version=10)

        print(trial_df.head(5))


# csv_directory = "final_split"
# csv_directory = "quick_split"
csv_directory = "DDD"

FINAL_BASIC_FILE = "data/DDD/listings_data_basic_XXX.csv"
FINAL_ENRICHED_FILE = "data/DDD/listings_data_enriched_XXX.csv"
FINAL_JSON_MODEL_FILE = "data/DDD/listings_data_jsonmodel_XXX.csv"
FINAL_JSON_META_FILE = "data/DDD/listings_data_jsonmeta_XXX.csv"
FINAL_RECENT_FILE = "data/source/df_listings.csv"
FINAL_RECENT_FILE_SAMPLE = "data/sample/df_listings_sample.csv"


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


def this_test_data(VERSION, test_data_only=False, drop_nulls=True):
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
        X_train, X_test, y_train, y_test = tt_split(VERSION, df)

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

        # np.savetxt(f"train_test/X_test{suffix}.csv", X_test[:20], delimiter=",")
        np.savetxt(f"train_test/X_test{suffix}.csv", X_test, delimiter=",")
        # np.savetxt(f"train_test/y_test{suffix}.csv", y_test[:20], delimiter=",")
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

def get_columns(version: int) -> pd.DataFrame:
    version_number = int(version)

    if version_number == 2:
        booleans = []
        floats = ['bedrooms', 'bathrooms', 'nearestStation', 'location.latitude', 'location.longitude']
        categories = ['tenure.tenureType']
        custom, wildcard = [],[]

    elif version_number == 3 or version_number == 4:
        booleans = []
        floats = ['bedrooms', 'bathrooms', 'nearestStation', 'latitude_deviation', 'longitude_deviation']
        categories = ['tenure.tenureType']
        custom, wildcard = [],[]

    elif version_number <= 6:
        booleans = []
        floats = ['bedrooms', 'bathrooms', 'nearestStation', 'location.latitude', 'location.longitude',
                  'latitude_deviation', 'longitude_deviation']
        categories = ['tenure.tenureType']
        custom, wildcard = [],[]

    elif version_number == 7:
        booleans = []
        floats = ['bedrooms', 'bathrooms', 'nearestStation', 'location.latitude', 'location.longitude',
                  'latitude_deviation', 'longitude_deviation']
        categories = ['tenure.tenureType']
        custom = ['listingHistory.listingUpdateReason']
        wildcard = []
    elif version_number == 8:
        booleans = []
        floats = ['bedrooms', 'bathrooms', 'nearestStation', 'location.latitude', 'location.longitude',
                  'latitude_deviation', 'longitude_deviation', 'keyFeatures']
        categories = ['tenure.tenureType']
        custom = ['listingHistory.listingUpdateReason']
        wildcard = []
    elif version_number in [9,10,11,12]:
        booleans = []
        floats = ['bedrooms', 'bathrooms', 'nearestStation', 'location.latitude', 'location.longitude',
                  'latitude_deviation', 'longitude_deviation']
        categories = ['tenure.tenureType']
        custom = [] #['reduced_features']
        wildcard = ['feature__']

    else:
        raise ValueError(f'no columns data available for version {version}')

    columns = []
    columns.extend(booleans)
    columns.extend(floats)
    columns.extend(categories)
    columns.extend(custom)
    #columns.extend(wildcard)

    return (columns, booleans, floats, categories, custom, wildcard)


def add_supplements(property_dataset, version):
    version_number = int(version)

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

    property_dataset['nearestStation'] = property_dataset['nearestStations'].apply(stations, args=['any'])
    property_dataset['nearestTram'] = property_dataset['nearestStations'].apply(stations, args=['TRAM'])
    property_dataset['nearestUnderground'] = property_dataset['nearestStations'].apply(stations,
                                                                                       args=['LONDON_UNDERGROUND'])
    property_dataset['nearestOverground'] = property_dataset['nearestStations'].apply(stations,
                                                                                      args=['overground'])

    if version_number in [9,10,11]:
        dff, aggregate_features = aggregate_keyFeatures(property_dataset['keyFeatures'])
        property_dataset['reduced_features'] = property_dataset['keyFeatures'].apply(reduce_keyFeatures, args=[dff.index,
                                                                                                           get_exploding_features(
                                                                                                               version_number)])

    return property_dataset




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


def aggregate_keyFeatures(key_features_column):
    from string import punctuation

    aggregated_features = []

    def clean(text):
        clean_each = [x.replace('*', '') for x in text]
        clean_each = [x.lstrip('-') for x in clean_each]
        clean_each = [x.replace('\t', '') for x in clean_each]
        clean_each = [x.lstrip('-') for x in clean_each]
        clean_each = [x.lstrip('-') for x in clean_each]
        clean_each = [x.lstrip('•') for x in clean_each]
        return clean_each

    for each_string in key_features_column:
        each = ast.literal_eval(each_string)
        # print("each\n", each)

        clean_each = clean(each)

        aggregated_features.extend(clean_each)
    ar_unique, i = np.unique(aggregated_features, return_counts=True)
    print(ar_unique, i)
    print(len(ar_unique), len(i))
    comb = np.vstack((ar_unique, i))
    dff = pd.DataFrame(comb.T, columns=['feature', 'occurrences'])
    dff['occurrences'] = pd.to_numeric(dff['occurrences'], 'coerce').dropna().astype(int)

    dff.sort_values('occurrences', ascending=False, inplace=True)
    dff.set_index('feature', inplace=True)

    return dff, aggregated_features


def reduce_keyFeatures(key_features_string, full_features, max_features):
    features_list = ast.literal_eval(key_features_string)

    allowed_features = full_features[:max_features]

    filtered_features_list = [x for x in features_list if x in allowed_features]
    if len(filtered_features_list) > 0:
        if False:
            print(f"{features_list}\n==>   {filtered_features_list}")

    return filtered_features_list


def get_exploding_features(version_number):
    if version_number < 9:
        EXPLODING_FEATURES = 0
    elif version_number == 9:
        EXPLODING_FEATURES = 10
    elif version_number == 10 or version_number == 11:
        EXPLODING_FEATURES = 50
    else:
        raise ValueError(f'missing parameter: version_number')

    return EXPLODING_FEATURES


def set_csv_directory(update_csv_directory):
    global csv_directory
    csv_directory = update_csv_directory

if __name__ == '__main__':
    test_module()
