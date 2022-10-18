from termcolor import colored

from functions_20221018_B import set_csv_directory, get_combined_dataset
from functions_20221018_B import add_supplements, tidy_dataset, feature_engineer

import pandas as pd

VERSION='03'

filename = f'df_listings_v{VERSION}.csv'
df_pathname_raw = f'../../data/source/{filename}'
df_pathname_tidy = f'../../data/final/{filename}'

LABEL = 'Price'

booleans = []
floats = ['bedrooms', 'bathrooms', 'nearestStation', 'latitude_deviation2', 'longitude_deviation2']
categories = ['tenure.tenureType']

columns = []
columns.extend(booleans)
columns.extend(floats)
columns.extend(categories)


cutdown_rows = 0


def build_dataset_versioned(version:int, folder_prefix='../../'):

    columns = []
    columns.extend(booleans)
    columns.extend(floats)
    columns.extend(categories)

    print(colored(f"features", "blue"), "-> ", columns)
    columns.insert(0, LABEL)
    print(colored(f"label", "green", None, ['bold']), "-> ", LABEL)

    set_csv_directory('final_split')

    print(f"VERSION {VERSION}: creating new source data.")
    retrieval_type = 'scratch'
    print(f'starting to get {retrieval_type} data...')
    df = get_combined_dataset(HOW='inner', early_duplicates=True, folder_prefix=folder_prefix)
    print(f'finished getting {retrieval_type} data!')

    # retrieval_type = 'RAW'
    # print(f'starting to save {retrieval_type} data...')
    # df.to_csv(df_pathname_raw)
    # print(f'finished saving {retrieval_type} data!')


    retrieval_type = 'RAW'
    print(f'adding supplements...')
    df = add_supplements(df)
    print(f'starting to save {retrieval_type} data...')
    df.to_csv(df_pathname_raw)
    print(f'finished saving {retrieval_type} data!')

    df = tidy_dataset(df, version=int(VERSION))
    df = feature_engineer(df, version=int(VERSION))

    df = df[columns]

    retrieval_type = 'TIDY'
    print(f'starting to save {retrieval_type} data...')
    df.to_csv(df_pathname_tidy)
    print(f'finished saving {retrieval_type} data!')


build_dataset_versioned(2)