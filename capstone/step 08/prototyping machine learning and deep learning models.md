## Introduction

I am using machine learning to predict the price of London properties.

I am using regression ML models for my project as I will be predicting continuous values.

I have looked at a number of models. Some were quickly trialled and discarded as being unsuitable for the task.
* Linear Regression in particular was shown to be ill-suited for training my model with my dataset and making predictions.
* Decision Trees was discarded quickly in favour of Random Forests, to allow for diversity of feature choice and redecing the degree of overfitting which is common with Decision Trees
* Support Vector machines were not considered after initial study, as they 

Other models were initially seen as unsuitable, but upon further examination brought back into consideration:
* Neural networks were originally assessed as ill-performing on the task. It transpired that poor results were due to lack of processing power which preventing the model from training long enough to produce good results.
    * As a result of this, the project code was refactored to allow the model to be proof-of-concept trained on a local high-performance machine, and then train properly on a long-running cloud machine which can allow deeper training to occur.
    * The neural network prototype is therefore submitted with the caveat that it is a proof-of-concept which will be enhanced further in order to achieve more accurate results

This left a few good contender models for the project:
* K Nearest Neighbours (regression)
* 
* 
* 
* 
* 
## Prototype Models

### Machine Learning Prototype
here: [Root Link](/process/E_train_model/it06_00_submission_prototype_model_capstone_step8_A_ML)

For my Machine Learning prototype I am submitting my catboost model.

### Deep Learning Prototype

I have trained a neural network which predict property prices.

here: [Root Link](/process/E_train_model/it06_00_submission_prototype_model_capstone_step8_B_DL)


## Appendix


### Dataset Used at this stage in the project
here: [Root Link](/data/final/df_listings_v06.csv)

### Short Previewable version
here: [Root Link](/data/sample/df_listings_v06_sample.csv)



### Python files for common functionality

[General code](/functions_0__common_20221116.py)
[Code related to assembling data](/functions_b__get_the_data_20221116.py)
[Code related to preparing and cleansing data](/functions_d1__prepare_cleanse_data_20221116.py)
[Code related to transforming and enriching data](/functions_d2__transform_enrich_data_20221116.py)
[Code related to preparing and storing data](/functions_d3__prepare_store_data_20221116.py)
[Code related to training models](/functions_d3__prepare_store_data_20221116.py)
[Code related to evaluating results](/functions_f_evaluate_model_20221116.py)
