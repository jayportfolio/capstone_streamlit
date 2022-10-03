import random

import streamlit as st
import pickle
from functions import this_test_data, this_df, build_model

import numpy as np

st.set_option('deprecation.showfileUploaderEncoding', False)

df, X_test, y_test = None, None, None
rand_index = -1

ALGORITHM = 'Decision Tree'


def main():
    global X_test, y_test, rand_index

    st.markdown(
        "<h1 style='text-align: center; color: White;background-color:#e84343'>London Property Prices Predictor</h1>",
        unsafe_allow_html=True)
    st.markdown(
        "<h3 style='text-align: center; color: Black;'>Insert your property parameters here, or choose a random pre-existing property.</h3>",
        unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: Black;'>Sub heading here</h4>",
                unsafe_allow_html=True)

    st.sidebar.header("What is this Project about?")
    st.sidebar.markdown(
        "This is a Web app that would predict the price of a London property based on parameters.")
    st.sidebar.header("Sidebar Options")
    include_nulls = st.sidebar.checkbox('include rows with any nulls ')
    if st.sidebar.button('Purge everything'):
        st.sidebar.error("I haven't added this functionality yet")
        # importing the os Library
        import os

        for deletable_file in [
            'X_test.csv', 'X_test_no_nulls.csv', 'X_train.csv', 'X_train_no_nulls.csv',
            'y_test.csv', 'y_test_no_nulls.csv', 'y_train.csv', 'y_train_no_nulls.csv',
            'model_Decision Tree.pkl',
            'model_Deep Neural Network.pkl',
            'model_HistGradientBoostingRegressor.pkl',
            'model_Linear Regression.pkl',
        ]:
            # checking if file exist or not
            if (os.path.isfile(deletable_file)):

                # os.remove() function to remove the file
                os.remove(deletable_file)

                # Printing the confirmation message of deletion
                print("File Deleted successfully:", deletable_file)
            else:
                print("File does not exist:", deletable_file)
            # Showing the message instead of throwig an error

    alg = ['Decision Tree', 'Linear Regression', 'Deep Neural Network', 'Linear Regression (Keras)',
           'HistGradientBoostingRegressor']
    ALGORITHM = st.selectbox('Which algorithm?', alg)

    try:
        model = pickle.load(open(f'model_{ALGORITHM}.pkl', 'rb'))
        # raise ValueError
    except:
        model = build_model(ALGORITHM, drop_nulls=~include_nulls)
        with open(f'model_{ALGORITHM}.pkl', 'wb') as f:
            pickle.dump(model, f)

    manual_parameters = st.checkbox('Use manual parameters instead of sample')
    if not manual_parameters:
        X_test, y_test = this_test_data(test_data_only=True)
        test_size = len(y_test)

        if st.button('randomise!'):
            rand_index = random.randint(0, test_size - 1)
            inputs = [X_test[rand_index]]

            random_instance = inputs
            np.savetxt("random_instance.csv", random_instance, delimiter=",")
            st.text(f'sample variables ({rand_index}): {inputs[0]}')
            st.text(f'Expected prediction: {y_test[rand_index]}')

            expected = y_test[rand_index]
            np.savetxt("random_instance.csv", random_instance, delimiter=",")
            random_instance_plus = [rand_index, expected]
            random_instance_plus.extend(random_instance[0])
            print("random_instance_plus:", random_instance_plus)
            np.savetxt("random_instance_plus.csv", [random_instance_plus], delimiter=",")

    else:
        lati = st.slider("Input Your latitude", 51.00, 52.00)
        longi = st.slider("Input your longitude", -0.5, 0.3)
        beds = st.slider("Input number of bedrooms", 0, 6)
        baths = st.slider("Input number of bathrooms", 0, 6)

        inputs = [[lati, longi, beds, baths]]

    if st.button('Predict'):

        if not manual_parameters:
            try:
                random_instance_plus = np.loadtxt("random_instance_plus.csv", delimiter=",")
                rand_index = random_instance_plus[0]
                expected = random_instance_plus[1]
                inputs = [random_instance_plus[2:]]
            except:
                # raise ValueError()
                rand_index = random.randint(0, test_size - 1)
                inputs = [X_test[rand_index]]
                random_instance = inputs
                expected = y_test[rand_index]
                np.savetxt("random_instance.csv", random_instance, delimiter=",")
                random_instance_plus = [rand_index, expected]
                random_instance_plus.extend(random_instance)
                print("random_instance_plus:",random_instance_plus)
                np.savetxt("random_instance_plus.csv", random_instance_plus, delimiter=",")

            st.text(f'sample variables ({rand_index}): {inputs[0]}')
            st.text(f'Expected prediction: {expected}')

        print("inputs:", inputs)

        result = model.predict(inputs)
        updated_res = result.flatten().astype(float)
        st.success('The predicted price for this property is {}'.format(updated_res))

    if st.checkbox('Show dataframe'):
        df = this_df()
        st.write(df)

    if st.checkbox('Get multiple predictions'):
        st.write(model.predict(X_test).flatten())

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
