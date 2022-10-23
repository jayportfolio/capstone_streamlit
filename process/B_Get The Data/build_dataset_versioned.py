from termcolor import colored

#from functions_20221018_C import set_csv_directory, get_combined_dataset, get_columns
from functions_20221021 import set_csv_directory, get_combined_dataset, get_columns
#from functions_20221018_C import add_supplements, tidy_dataset, feature_engineer, preprocess
from functions_20221021 import add_supplements, tidy_dataset, feature_engineer, preprocess

import pandas as pd

cutdown_rows = 0


def main():
    build_dataset_versioned(6)
    print("Finished!!")
    

def build_dataset_versioned(version_number:int, folder_prefix='../../'):
    VERSION = '0' + str(version_number)

    filename = f'df_listings_v{VERSION}.csv'
    df_pathname_raw = f'../../data/source/{filename}'
    df_pathname_untidy = f'../../data/source/untidy_{filename}'
    df_pathname_tidy = f'../../data/final/{filename}'

    LABEL = 'Price'

    columns, booleans, floats, categories = get_columns(version=version_number)

    print(colored(f"features", "blue"), "-> ", columns)
    columns.insert(0, LABEL)
    print(colored(f"label", "green", None, ['bold']), "-> ", LABEL)

    set_csv_directory('final_split')

    print(f"VERSION {VERSION}: creating new source data.")
    retrieval_type = 'scratch'
    print(f'starting to get {retrieval_type} data...')
    df = get_combined_dataset(HOW='inner', early_duplicates=True, folder_prefix=folder_prefix)
    print(f'finished getting {retrieval_type} data!')

    retrieval_type = 'RAW'
    print(f'adding supplements...')
    df = add_supplements(df)
    print(f'starting to save {retrieval_type} data...')
    df.to_csv(df_pathname_raw)
    print(f'finished saving {retrieval_type} data!')

    retrieval_type = 'UNTIDY'
    df_copy_for_untidy = df.copy()
    print(f'starting to save {retrieval_type} data...')
    df_copy_for_untidy = feature_engineer(df_copy_for_untidy, version=version_number)
    df_copy_for_untidy = df_copy_for_untidy[columns]
    df_copy_for_untidy.to_csv(df_pathname_untidy)
    print(f'finished saving {retrieval_type} data!')


    df = tidy_dataset(df, version=version_number)
    df = feature_engineer(df, version=version_number)
    df = preprocess(df, version=version_number)

    df = df[columns]

    retrieval_type = 'TIDY'
    print(f'starting to save {retrieval_type} data...')
    df.to_csv(df_pathname_tidy)
    print(f'finished saving {retrieval_type} data!')

    print("Finished!!")


if __name__ == '__main__':
    main()