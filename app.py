import random

import streamlit as st
import pickle
from functions import this_test_data, this_df, build_model

import numpy as np

st.set_option('deprecation.showfileUploaderEncoding', False)

model = None

df, X_test, y_test = None, None, None
rand_index = -1

ALGORITHM = 'Decision Tree'


def main():
    global X_test, y_test, rand_index

    st.markdown(
        "<h1 style='text-align: center; color: White;background-color:#e84343'>London Property Prices Predictor</h1>",
        unsafe_allow_html=True)
    st.markdown(
        "<h3 style='text-align: center; color: Black;'>Insert yeur property parameters here, or choose a random pre-existing property.</h3>",
        unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: Black;'>Sub heading here</h4>",
                unsafe_allow_html=True)
    st.sidebar.header("What is this Project about?")
    st.sidebar.text(
        "This is a Web app that would predict the price of a London property based on parameters.")
    st.sidebar.header("Sidebar header?")
    st.sidebar.text(
        "Info goes here")

    alg = ['Decision Tree', 'Linear Regression', 'Deep Neural Network', 'Linear Regression (Keras)']
    ALGORITHM = st.selectbox('Which algorithm?', alg)

    try:
        model = pickle.load(open(f'model_{ALGORITHM}.pkl', 'rb'))
        # raise ValueError
    except:
        model = build_model(ALGORITHM)
        with open(f'model_{ALGORITHM}.pkl', 'wb') as f:
            pickle.dump(model, f)

    manual_parameters = st.checkbox('Use manual parameters instead of sample')
    if not manual_parameters:
        X_test, y_test = this_test_data(test_data_only=True)
        test_size = len(y_test)

        if st.button('randomise!'):
            rand_index = random.randint(0, test_size)
            inputs = [X_test[rand_index]]

            random_instance = inputs
            np.savetxt("random_instance.csv", random_instance, delimiter=",")
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

        if not manual_parameters:
            try:
                random_instance = np.loadtxt("random_instance.csv", delimiter=",")
                inputs = [random_instance]
            except:
                rand_index = random.randint(0, test_size)
                inputs = [X_test[rand_index]]
                random_instance = inputs
                np.savetxt("random_instance.csv", random_instance, delimiter=",")

            st.text(f'sample variables ({rand_index}): {inputs[0]}')
            st.text(f'Expected prediction: {y_test[rand_index]}')

        result = model.predict(inputs)
        updated_res = result.flatten().astype(float)
        st.success('The predicted price for this property is {}'.format(updated_res))

    if st.checkbox('Show dataframe'):
        df = this_df()
        st.write(df)

    if st.checkbox('Show predictions and accuracy'):
        X_train, X_test, y_train, y_test = this_test_data()
        acc = model.score(X_test, y_test)
        st.write('Accuracy: ', acc)
        pred_dtc = model.predict(X_test)
        # cm_dtc = confusion_matrix(y_test, pred_dtc)
        # st.write('Confusion matrix: ', cm_dtc)
        st.write('Predictions: ', pred_dtc)


if __name__ == '__main__':
    main()
