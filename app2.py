import random

import streamlit as st
import pickle
import pandas as pd
from functions import get_combined_dataset

st.set_option('deprecation.showfileUploaderEncoding', False)

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

ROW_LIMIT = 300
LABEL = 'Price'

dtc = None

df, X_test, y_test = None, None, None
rand_index = -1


def this_test_data():
    import numpy as np

    try:
        #X_test = np.loadtxt("X_test.csv", delimiter=",", dtype=str)
        X_test = np.loadtxt("X_test.csv", delimiter=",")
        y_test = np.loadtxt("y_test.csv", delimiter=",")
    except:

        # global X_test, y_test
        # if X_test is not None and y_test is not None:
        #     return X_test, y_test

        df = this_df()
        features = df[df.columns[:-1]].values
        labels = df[LABEL].values
        X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=1)

        np.savetxt("X_test.csv", X_test[:20], delimiter=",")
        np.savetxt("y_test.csv", y_test[:20], delimiter=",")

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


def build_model():
    # global df, dtc
    global X_test, y_test, dtc

    df = this_df()
    features = df[df.columns[:-1]].values
    labels = df[LABEL].values
    X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=1)

    dtc = DecisionTreeRegressor()
    # dtc = HistGradientBoostingClassifier()
    dtc.fit(X_train, y_train)

    return dtc


try:
    dtc = pickle.load(open('new_model2.pkl', 'rb'))
    # raise ValueError
except:
    dtc = build_model()
    with open('new_model2.pkl', 'wb') as f:
        pickle.dump(dtc, f)


def main():
    global X_test, y_test, rand_index

    st.markdown(
        "<h1 style='text-align: center; color: White;background-color:#e84343'>Graduate Admission Predictor</h1>",
        unsafe_allow_html=True)
    st.markdown(
        "<h3 style='text-align: center; color: Black;'>Drop in The required Inputs and we will do  the rest.</h3>",
        unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: Black;'>Submission for The Python Week</h4>",
                unsafe_allow_html=True)
    st.sidebar.header("What is this Project about?")
    st.sidebar.text(
        "It a Web app that would help the user in determining whether they will get admission in a Graduate Program or not.")
    st.sidebar.header("What tools where used to make this?")
    st.sidebar.text(
        "The Model was made using a dataset from Kaggle along with using Kaggle notebooks to train the model. We made use of Sci-Kit learn in order to make our Linear Regression Model.")

    if st.checkbox('Use sample property'):
        X_test, y_test = this_test_data()
        test_size = len(y_test)

        if st.button('randomise!'):
            rand_index = random.randint(0, test_size)

        inputs = [X_test[rand_index]]
        st.text(f'sample variables ({rand_index}): {inputs[0]}')
        st.text(f'Expected prediction: {y_test[rand_index]}')
    else:
        lati = st.slider("Input Your latitude", 51.00, 52.00)
        longi = st.slider("Input your longitude", -0.5, 0.3)
        # toefl = st.slider("Input your TOEFL Score", 0, 120)
        # research = st.slider("Do You have Research Experience (0 = NO, 1 = YES)", 0, 1)
        # uni_rating = st.slider("Rating of the University you wish to get in on a Scale 1-5", 1, 5)

        # inputs = [[lati, longi, toefl, research, uni_rating]]
        inputs = [[lati, longi]]

    if st.button('Predict'):
        rand_index = random.randint(0, test_size)

        inputs = [X_test[rand_index]]
        st.text(f'sample variables ({rand_index}): {inputs[0]}')
        st.text(f'Expected prediction: {y_test[rand_index]}')

        result = dtc.predict(inputs)
        updated_res = result.flatten().astype(float)
        st.success('The predicted price for this property is {}'.format(updated_res))

    if st.checkbox('Show dataframe'):
        df = this_df()
        st.write(df)

    if st.checkbox('Show predictions and accuracy'):
        df = this_df()
        features = df[df.columns[:-1]].values
        labels = df[LABEL].values
        X_train, X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=1)
        acc = dtc.score(X_test, y_test)
        st.write('Accuracy: ', acc)
        pred_dtc = dtc.predict(X_test)
        # cm_dtc = confusion_matrix(y_test, pred_dtc)
        # st.write('Confusion matrix: ', cm_dtc)
        st.write('Predictions: ', pred_dtc)


if __name__ == '__main__':
    main()
