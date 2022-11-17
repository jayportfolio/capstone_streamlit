There are many previous attempts to predict property prices within a geographic region, in a generalised sense.

# Research into previous attempts at property price prediction
### The Classic Example - Boston House Prices

A staple amongst machine learning beginners, this dataset is used to teach machine learning techniques for regression problems.

There have been many prior attempts to create an optimised prediction model for this specific problem.

Examples:
* https://www.kaggle.com/code/mohaiminul101/boston-house-prices-linear-regression
* https://www.kaggle.com/datasets/fedesoriano/the-boston-houseprice-data
* https://www.kaggle.com/code/hugosjoberg/house-prices-prediction-using-keras

Techniques used to create prediction models include:
* Multiple linear regression
* XGBoost
* Decision Tree with Bagging (https://www.kaggle.com/code/pear2jam/bagging-hyperparameters/notebook)
* Neural Networks

### Research Papers


https://www.tandfonline.com/doi/pdf/10.1080/09599916.2020.1832558?needAccess=true

Advocates for the usage of RF (random forest) and GBM (gradient boosting machines) in pursuit of an optimal property price prediction model, especially in preference to SVM (support vector machines).


# Applying Prior Research to my dataset

I have decided to apply two of the researched notebooks I found to my dataset 

### Application 1:

This uses Linear Regression to predict property prices for a Kaggle challenge.

#### The original work
https://colab.research.google.com/drive/1DXkpo9PmH9_HiCSz9NQlZ9vGQtMIYqmF

#### Applying the original work to my dataset
[Application to my dataset](./06_01_02__apply_researched_solution_to_my_dataset.ipynb)


### Application 2:

This uses a Neural Network to predict property prices for the same Kaggle challenge. 

#### The original work
https://www.kaggle.com/code/hugosjoberg/house-prices-prediction-using-keras challenge to my dataset

#### Applying the original work to my dataset
[Application to my dataset](./06_02_02__apply_house-prices-prediction-using-keras.ipynb)


## Lessons learned

I learned that Linear Regression performs poorly on data of this sort, and that Neural Networks appear to work better. 
I am more aware that it is neccessary to explore a variety of models to get the best results, and that it may be necessary to explore the hyperparameters of a model to improve upon it further. 