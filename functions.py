import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from tensorflow_estimator.python.estimator.canned.linear import LinearRegressor

ROW_LIMIT = 300

#df, X_test, y_test = None, None, None
df = None

FINAL_ROW_COUNT = 100

ROW_LIMIT = 300
AFFORDABLE_PRICE = 400000
NA_THRESH = 100

DROP_COLS = ['businessForSale', 'affordableBuyingScheme', 'status.published', 'address.deliveryPointId',
             'location.showMap', 'misInfo.branchId', 'misInfo.premiumDisplayStampId', 'misInfo.brandPlus']

#LABEL = 'variety'
#LABEL = 'Affordable'
LABEL = 'Price'

FINAL_BASIC_FILE = "../../data/final_split/listings_data_basic_XXX.csv"
FINAL_ENRICHED_FILE = "../../data/final_split/listings_data_enriched_XXX.csv"
FINAL_JSON_MODEL_FILE = "../../data/final_split/listings_data_jsonmodel_XXX.csv"
FINAL_JSON_META_FILE = "../../data/final_split/listings_data_jsonmeta_XXX.csv"

FINAL_BASIC_FILE = "data/final_split/listings_data_basic_XXX.csv"
FINAL_ENRICHED_FILE = "data/final_split/listings_data_enriched_XXX.csv"
FINAL_JSON_MODEL_FILE = "data/final_split/listings_data_jsonmodel_XXX.csv"
FINAL_JSON_META_FILE = "data/final_split/listings_data_jsonmeta_XXX.csv"
FINAL_RECENT_FILE = "data/final_split/data.csv"


def pre_tidy_dataset(property_dataset):
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

    return property_dataset

def get_combined_dataset(HOW, early_duplicates, row_limit=None, verbose=False):
  df_list = get_df(FINAL_BASIC_FILE)
  # df_indiv = get_df(FINAL_ENRICHED_FILE, testing)
  df_meta = get_df(FINAL_JSON_META_FILE)
  # df_json1 = get_df(LISTING_JSON_MODEL_FILE, on_bad_lines='warn')  # EDIT 29-06-2022: There are bid listings and regular listings. I scrape them seporately and join them here.
  # df_json = get_df(LISTING_JSON_MODEL_FILE)
  df_json = get_df(FINAL_JSON_MODEL_FILE)

  df_meta['id_copy'] = df_meta['id_copy'].astype(int)
  df_json['id'] = df_json['id'].astype(int)
  df_list.set_index(['ids'], inplace=True)
  # df_indiv.set_index(['ids'], inplace=True)
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
        .join(df_meta, how='inner', lsuffix='', rsuffix='_meta')  # \
    # .join(df_indiv, on='ids', how='inner', lsuffix='', rsuffix='_listing') \
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


def this_test_data(test_data_only = False):
    import numpy as np

    try:
        if not test_data_only:
            X_train = np.loadtxt("X_train.csv", delimiter=",")
            y_train = np.loadtxt("y_train.csv", delimiter=",")

        X_test = np.loadtxt("X_test.csv", delimiter=",")
        y_test = np.loadtxt("y_test.csv", delimiter=",")
    except:
        df = this_df()
        features = df[df.columns[:-1]].values
        labels = df[LABEL].values
        X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=1)

        if not test_data_only:
            np.savetxt("X_train.csv", X_train[:20], delimiter=",")
            np.savetxt("y_train.csv", y_train[:20], delimiter=",")

        np.savetxt("X_test.csv", X_test[:20], delimiter=",")
        np.savetxt("y_test.csv", y_test[:20], delimiter=",")

    if not test_data_only:
        return X_train, X_test, y_train, y_test

    return X_test, y_test


def this_df():
    global df

    if df is not None:
        return df

    df = get_combined_dataset('inner2', True, row_limit=ROW_LIMIT)
    df = df[['location.latitude', 'location.longitude', 'Price']]
    df['Price'] = pd.to_numeric(df['Price'], 'coerce').dropna().astype(int)
    # for each in ['bedrooms', 'location.latitude', 'location.longitude']:
    for each in ['location.latitude', 'location.longitude']:
        df[each] = pd.to_numeric(df[each], 'coerce').dropna().astype(float)
    return df


def build_model(algorithm):
    X_train, X_test, y_train, y_test = this_test_data()

    if algorithm == 'Decision Tree':
        model = DecisionTreeRegressor()
        # dtc = HistGradientBoostingClassifier()
        model.fit(X_train, y_train)
    elif algorithm == 'Linear Regression':
        model = LinearRegression()
        model.fit(X_train, y_train)
    elif algorithm == 'Deep Neural Network':

        import tensorflow as tf

        from tensorflow import keras
        from tensorflow.keras import layers

        print(tf.__version__)

        def build_and_compile_model(norm):
            model = keras.Sequential([
                norm,
                layers.Dense(64, activation='relu'),
                layers.Dense(64, activation='relu'),
                layers.Dense(1)
            ])

            model.compile(loss='mean_absolute_error',
                          optimizer=tf.keras.optimizers.Adam(0.001))
            return model

        normalizer = tf.keras.layers.Normalization(axis=-1)

        dnn_model = build_and_compile_model(normalizer)
        #print(dnn_model.summary())

        #% % time
        history = dnn_model.fit(
            X_train, #train_features,
            y_train, #train_labels,
            validation_split=0.2,
            verbose=0, epochs=100)

        def plot_loss(history):
            import matplotlib.pyplot as plt
            plt.plot(history.history['loss'], label='loss')
            plt.plot(history.history['val_loss'], label='val_loss')
            plt.ylim([0, 10])
            plt.xlabel('Epoch')
            plt.ylabel('Error [MPG]')
            plt.legend()
            plt.grid(True)

        plot_loss(history)

        #test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)
        print(dnn_model.evaluate(X_train, y_train, verbose=0))

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

        #%%time
        history = linear_model.fit(
            X_train, #train_features,
            y_train, #train_labels,
            epochs=100,
            # Suppress logging.
            verbose=0,
            # Calculate validation results on 20% of the training data.
            validation_split=0.2)

        def plot_loss(history):
            import matplotlib.pyplot as plt
            plt.plot(history.history['loss'], label='loss')
            plt.plot(history.history['val_loss'], label='val_loss')
            plt.ylim([0, 10])
            plt.xlabel('Epoch')
            plt.ylabel('Error [MPG]')
            plt.legend()
            plt.grid(True)

        plot_loss(history)

        model = linear_model

    elif algorithm == 'Linear Regression (Keras)':
        model = LinearRegressor()
        model.fit(X_train, y_train)
    else:
        raise ValueError(algorithm)

    return model
