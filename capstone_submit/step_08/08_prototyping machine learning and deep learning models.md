## Introduction

### About the Project
I am using machine learning to predict the price of London properties.

### About the Models

I am using regression ML models for my project as I will be predicting continuous values.

I have looked at a number of models. Some were quickly trialled and discarded as being unsuitable for the task.
* Logistic Regression was immediately discarded since I am solving a regression problem rather than a classification problem.
* Linear Regression in particular was shown to be ill-suited for training my model with my dataset and making predictions.
    * The relationship between the features and the label is more nuanced than strictly linear, making it difficult for Linear Regression to extract accurate predictions.
        * EDA makes it clear some features are polynomial (latitude and longitude are quadratic, because distance from centre is linear). Using polynomial linear regression could be used to account for this but would add to complexity and risk overfitting without careful hyperparameter tuning.
    * Many of the features are correlated, which makes Linear Regression especially unsuitable
* Decision Trees was discarded quickly in favour of Random Forests, to allow for diversity of feature choice and redecing the degree of overfitting which is common with Decision Trees

Other models were initially seen as unsuitable, but upon further examination brought back into consideration:
* Neural networks were originally assessed as ill-performing on the task. It transpired that poor results were due to lack of processing power which preventing the model from training long enough to produce good results.
    * As a result of this, the project code was refactored to allow the model to be proof-of-concept trained on a local high-performance machine, and then train properly on a long-running cloud machine which can allow deeper training to occur.
    * The neural network prototype is therefore submitted with the caveat that it is a proof-of-concept which will be enhanced further in order to achieve more accurate results

This left a few good contender models for the project:
* KNN (K Nearest Neighbours) for regression
  * This model works well.
    * One theory of why this model works so well for this data is because of its clustering-style approach to prediction, which mimics the human subliminal tendency to assume that properties which can be described the same way are worth the same price. 
* Random Forests
  * This has the advantage of decision trees but suffers less from over-fitting.
  * It is less susceptible to outliers than some other models in my portfolio
  * If an appropriate implementation of Random Forests can be obtained, it could be used on data instances which have missing data
  * It also provides the ability to track feature importances it decided upon, which is in turn useful for next-iteration data exploration.  
* CatBoost 
  * This boosting algorithm capitalises upon any advantages it discovers from previous rounds of training, and allows less obvious traits in data to be exploited
  * The aggregation of many weak learners into a strong learner may produce the strongest model of all, if increasing numbers of features are used
  * It doesn't work well on sparse data, but that is not an issue for my dataset, because my data collection was oriented towards collecting extremely dense structured data
  * Like Random Forests, it allows calculated feature importance to be obtained. 
* XG Boost and LGB:
  * These are both great models, but at this stage surpassed by the catboost model.
  * I plan to continue investigating them both, but may discard them if CatBoost remains consistently superior
* Neural Networks and Deep Learning
  * Has the most potential to produce great predictive power if given enough computational power.
  * As such, remains in the mix for my final modelling solution.
  * May prove more consistently accurate if provided with more data or hypertuned to perform better.

### About the data
My data is split into training and testing sets.

The machine learning model uses RandomisedSearchCV to find the best hyperparameters, 
The training data is used for training, and cross-validation to prevent overfitting and find the best overall model.
The test data is used to evaluate the effectiveness of my trained model.


The deep learning model splits the data into training data, validation, and testing data 
The training data is split into training data and validation, and used to train the model over a series of epochs.
The test data is used to evaluate the effectiveness of my trained model.

## Prototype Models

### Machine Learning Prototype
[Machine Learning Notebook](../../process/E_train_model/iteration06/it06_00_submission_prototype_model_capstone_step8_A_ML.ipynb)

For my Machine Learning prototype I am submitting my KNN model.

### Deep Learning Prototype

I have trained a neural network which predict property prices.

[Neural Network notebook](../../process/E_train_model/iteration06/it06_00_submission_prototype_model_capstone_step8_B_DL.ipynb)

## Prototype Results

Prototyping has shown KNN to be one of the best performing models for predicting property prices using my data.

Neural Networks performs adequately but less well, and will require more investigation as to whether it is a viable model for this project.

[Prototype Results](../../results/summary_benchmark_v06.ipynb)

N.B. CatBoost also performs well, and will require further investigation to see if hypertuning, better/more data or other modifications can produce better results than KNN.

## Appendix


### Dataset Used at this stage in the project
[Dataset for ML/DL prototypes (version 6 of feature selection/feature engineering)](../../data/final/df_listings_v06.csv)

### Short Previewable version of Dataset
[Preview Dataset for ML/DL prototypes (version 6 of feature selection/feature engineering)](../../data/sample/df_listings_v06_sample.csv)



### Python files for common functionality

[Code related to assembling data](../../functions_b__get_the_data_20221116.py)

[Code related to preparing and cleansing data](../../functions_d1__prepare_cleanse_data_20221116.py)

[Code related to transforming and enriching data](../../functions_d2__transform_enrich_data_20221116.py)

[Code related to preparing and storing data](../../functions_d3__prepare_store_data_20221116.py)

[Code related to training models](../../functions_d3__prepare_store_data_20221116.py)

[Code related to evaluating results](../../functions_f_evaluate_model_20221116.py)

[General code](../../functions_0__common_20221116.py)

