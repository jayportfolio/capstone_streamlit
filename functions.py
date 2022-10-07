import json
import math

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

ROW_LIMIT = 3000
ROW_LIMIT = 0

# df, X_test, y_test = None, None, None
df = None

FINAL_ROW_COUNT = 100

AFFORDABLE_PRICE = 400000
NA_THRESH = 100

# LABEL = 'variety'
# LABEL = 'Affordable'
LABEL = 'Price'
# LABEL = 'Price'
floats = ['location.latitude', 'location.longitude', 'bedrooms', 'bathrooms',
          #'distance_to_any_train',
          #'sharePercentage',
          #'sharedOwnership.ownershipPercentage',
          'nearestStation','nearestTram','nearestUnderground','nearestOverground',
          ]
# floats = ['location.latitude', 'location.longitude']
# categories = ['borough']
successful_categories_300rows_20221004 = ['tenure.tenureType', 'analyticsProperty.soldSTC',
                                          'analyticsProperty.preOwned',
                                          'sharedOwnership.sharedOwnership', 'analyticsProperty.propertyType']
categories = ['tenure.tenureType',
              'analyticsProperty.soldSTC',
              'analyticsProperty.preOwned',
              'sharedOwnership.sharedOwnership',
              'analyticsProperty.propertyType',  # 'propertyType',
               'analyticsProperty.propertySubType',
               'borough',
              'analyticsProperty.priceQualifier',

              ]
# categories = []
FEATURES = floats.copy()
FEATURES.extend(categories)

DROP_COLS = ['businessForSale', 'affordableBuyingScheme', 'status.published', 'address.deliveryPointId',
             'location.showMap', 'misInfo.branchId', 'misInfo.premiumDisplayStampId', 'misInfo.brandPlus']


# csv_directory = "final_split"
csv_directory = "quick_split"
FINAL_BASIC_FILE = "data/%s/listings_data_basic_XXX.csv" % csv_directory
FINAL_ENRICHED_FILE = "data/%s/listings_data_enriched_XXX.csv" % csv_directory
FINAL_JSON_MODEL_FILE = "data/%s/listings_data_jsonmodel_XXX.csv" % csv_directory
FINAL_JSON_META_FILE = "data/%s/listings_data_jsonmeta_XXX.csv" % csv_directory
FINAL_RECENT_FILE = "data/source/df_listings.csv"
FINAL_RECENT_FILE_SAMPLE = "data/sample/df_listings_sample.csv"


def pre_tidy_dataset(property_dataset):

    #     df = df[~df.index.duplicated(keep='last')]

    property_dataset['Price'] = pd.to_numeric(property_dataset['Price'], 'coerce').dropna().astype(int)

    # do any necessary renames, and some preliminary feature engineering
    try:
        property_dataset = property_dataset.rename(index=str, columns={"Station_Prox": "distance_to_any_train"})
    except:
        pass

    # try:
    #     property_dataset['floorplan_count'] = property_dataset['floorplans'].apply(get_array_length)
    # except:
    #     pass

    try:
        # property_dataset['borough_name'] = property_dataset["borough"].str.extract("\('(.+)',")
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
        # property_dataset['borough_name'] = property_dataset["borough"].str.extract("\('(.+)',")
        # property_dataset['propertyType'] = property_dataset["analyticsProperty.propertyType"].str.extract("(.+) /")
        # property_dataset['propertyType'] = property_dataset["analyticsProperty.propertyType"].str.split("/")[0]
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

    # try:
    # except:
    #     pass

    # property_dataset['type'] = property_dataset[\"Description\"].str.extract(\"(house|apartment|flat|maisonette)\")
    # property_dataset['hold_type2'] = property_dataset[\"hold_type\"].str.replace(\"Tenure:\",\"\").str.strip()

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
        #print('stations')
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
                ['NATIONAL_TRAIN','LONDON_UNDERGROUND', 'LONDON_OVERGROUND'],
                ['NATIONAL_TRAIN', 'LONDON_UNDERGROUND'],
                ['NATIONAL_TRAIN', 'LIGHT_RAILWAY'],
                ['NATIONAL_TRAIN', 'TRAM'],
                ['NATIONAL_TRAIN', 'LONDON_OVERGROUND'],
                ['NATIONAL_TRAIN', 'TRAM','LONDON_OVERGROUND'],
                ['NATIONAL_TRAIN','LONDON_UNDERGROUND', 'LIGHT_RAILWAY'],
                ['NATIONAL_TRAIN','LONDON_UNDERGROUND', 'TRAM'],
                ['NATIONAL_TRAIN', 'LONDON_UNDERGROUND', 'LIGHT_RAILWAY', 'LONDON_OVERGROUND'],
            ]:
                print(station['types'], ":", station_list)

            if requested_type == 'any':
                # print(station)
                return station['distance']
            elif requested_type in station['types']:
                return station['distance']
            elif requested_type == "overground" and ('NATIONAL_TRAIN' in station['types'] or 'LONDON_OVERGROUND' in station['types'] or 'LIGHT_RAILWAY' in station['types']):
                return station['distance']
            elif requested_type == "underground combined" and 'LONDON_UNDERGROUND' in station['types'] and len(station['types']) > 1:
                return station['distance']
            else:
                pass

        #return 99
        return None

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


def get_combined_dataset(HOW, early_duplicates, row_limit=None, verbose=False):
    df_list = get_df(FINAL_BASIC_FILE)
    # df_indiv = get_df(FINAL_ENRICHED_FILE, testing)
    df_indiv = get_df(FINAL_ENRICHED_FILE)
    df_meta = get_df(FINAL_JSON_META_FILE)
    # df_json1 = get_df(LISTING_JSON_MODEL_FILE, on_bad_lines='warn')  # EDIT 29-06-2022: There are bid listings and regular listings. I scrape them seporately and join them here.
    # df_json = get_df(LISTING_JSON_MODEL_FILE)
    df_json = get_df(FINAL_JSON_MODEL_FILE)

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
            .join(df_meta, how='inner', lsuffix='', rsuffix='_meta') #\
            #.join(df_indiv, how='inner', lsuffix='', rsuffix='_listing')
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

    df_original.iloc[:20].to_csv('data/source/df_source_full_sample.csv')
    df_original.to_csv('data/source/df_source_full.csv')

    if row_limit and row_limit > 0:
        # return df_original[:row_limit]
        return df_original.sample(n=row_limit)
    return df_original


def get_df(file):
    df_array = []
    for n in range(10):
        prefix = '00' if n <= 10 else '0'

        filename = file.replace('XXX', prefix + str(n))

        from os.path import exists
        file_exists = exists(filename)

        if file_exists:
            if n == 0:
                merged_df = pd.read_csv(filename, on_bad_lines='skip')
            else:
                # each_df = pd.read_csv(filename, on_bad_lines='skip')
                # df_array.append(each_df)
                merged_df = pd.concat([merged_df, pd.read_csv(filename, on_bad_lines='skip')])
                print(n)

            # print(f'error on {file} at split {prefix}')
        else:
            print(f'{n + 1} splits for {file}')
            break

    return merged_df


def this_test_data(test_data_only=False, drop_nulls=True):
    suffix = "_no_nulls" if drop_nulls else ""

    try:
        if not test_data_only:
            X_train = np.loadtxt(f"train_test/X_train{suffix}.csv", delimiter=",")
            y_train = np.loadtxt(f"train_test/y_train{suffix}.csv", delimiter=",")

        X_test = np.loadtxt(f"train_test/X_test{suffix}.csv", delimiter=",")
        y_test = np.loadtxt(f"train_test/y_test{suffix}.csv", delimiter=",")
    except:
        df = this_df()

        if drop_nulls:
            df.dropna(inplace=True)

        # xxxfeatures = df[df.columns[:-1]].values
        # features = df[FEATURES].values
        # labels = df[LABEL].values
        # X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=1)
        X_train, X_test, y_train, y_test = tt_split(df)

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

        np.savetxt(f"train_test/X_test{suffix}.csv", X_test[:20], delimiter=",")
        np.savetxt(f"train_test/y_test{suffix}.csv", y_test[:20], delimiter=",")

    if not test_data_only:
        return X_train, X_test, y_train, y_test

    return X_test, y_test


def tt_split(df):
    for column in categories:
        df = pd.concat([df, pd.get_dummies(df[column], prefix=column)], axis=1)
        df.drop([column], axis=1, inplace=True)  # now drop the original column (you don't need it anymore),

    # features = df[df.columns[:-1]].values
    features = df[df.columns[1:]].values
    # features = df[FEATURES].values
    labels = df[LABEL].values
    X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=1)
    return X_train, X_test, y_train, y_test


def this_df(row_limit=ROW_LIMIT):
    global df

    if df is not None:
        return df

    df = get_combined_dataset('inner2', True, row_limit=row_limit)
    df = pre_tidy_dataset(df)
    print("FEATURES:", FEATURES)

    columns = FEATURES.copy()
    columns.insert(0, LABEL)

    print('columns:', columns)
    print('LABEL:', LABEL)

    df = df[columns]
    df[LABEL] = pd.to_numeric(df[LABEL], 'coerce').dropna().astype(int)

    for each in floats:
        df[each] = pd.to_numeric(df[each], 'coerce').dropna().astype(float)

    df.to_csv(FINAL_RECENT_FILE, index=True)
    df.sample(20).to_csv(FINAL_RECENT_FILE_SAMPLE)
    return df
