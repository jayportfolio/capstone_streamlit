# Experimenting With Various Models

## Dataset

At this stage I am using iteration 06 of my dataset features.
<br>The dataset can be found
here: [Root Link](/data/final/df_listings_v06.csv)
or [Relative Link](../../data/final/df_listings_v06.csv)

## Model Training

I test on a variety of regression models, and use cross-validation via RandomGridSearchCV to optimize the
hyperparameters while ensuring my model does not suffer from overfitting.

## Compiled Models

Directory of trained models [Relative Link](../process/E_train%20model/iteration06/)

I have trained six models and recorded model performance and additional helpful metadata in a json file.
<br> I then summarise the data in a notebook and analyse to identify the best performance model for the data.

Notebooks:

* [Linear Regression (Ridge) Model](../process/E_train%20model/iteration06/it06_01_4__linear_regression_randomsearch__20221019.ipynb)
* [KNN Model](../process/E_train%20model/iteration06/it06_02_4__knn_randomsearch__20221106.ipynb)
* [Decision Tree Model](../process/E_train%20model/iteration06/it06_03_4__decision_tree_randomsearch__20221106.ipynb)
* [XGBoost Model](../process/E_train%20model/iteration06/it06_04_4__xgboost_randomsearch__20221018.ipynb)
* [Neural Network Model](../process/E_train%20model/iteration06/it06_05_4__neural_network_randomsearch__20221106.ipynb)
* [Random Forest Model](../process/E_train%20model/iteration06/it06_06_4__random_forest_randomsearch__20221106.ipynb)

## Summarisation

I look at multiple performance metrics to gauge model effectiveness:

* F2 score (primary metric and used for cross validation)
* MAE
* MSE
* RMSE
* in a future iteration I may additionally look at RMSLE: Root Mean Squared Logarithmic Error since it:
    * is a good metric if target has exponential growth
    * punishes underpredicted estimates more than overpredicted estimates, which is a particular problem for my data (
      where properties are incorrectly underpredicted)

### Comparison of performance

* [summary of performance](../../results/summary_benchmark_v06.ipynb)


| Item                      | Score (F2) | Approx Training Time |
|---------------------------|------------|----------------------|
| **KNN**                   | 0.719095   | 0.014735             |
| **XG Boost**              | 0.714029   | 4.448235             |
| Random Forest             | 0.63329    | 39.140589            |
| Decision Tree             | 0.498178   | 36.841502            |
| Neural Network            | 0.496803   | long (unknown)       |
| Linear Regression (Ridge) | 0.456885   | 0.095841             |

From the summarisation, I can see that the best performing models are (currently):

* K Nearest Neighbour
* XGBoost
  These will be the models I focus the majority of time on, for further hyperparameter tuning and deeper analysis. 