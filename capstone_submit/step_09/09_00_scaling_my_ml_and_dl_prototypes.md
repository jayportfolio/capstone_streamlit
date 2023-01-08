# Scaling my ML (and DL) Prototypes

## Dataset

I have scaled my models by increasing the numbers of features dramatically, by gathering additional features from the property description. I have used text analysis to find features which are known to be desirable in a property, and used them as dataset features

I made three new dataset versions by extracting popular property features:
* version 09: I added 10 features extracted from the property description
* version 10  I added 50 features extracted from property description
* version 11. In addition to all the features in version 10, I used a separate text extraction technique to get even more common property features from the property description, bringing the total number of features to 80.

<br>The version 11 dataset can be found
here: [Root Link](/data/sample/df_listings_v11_sample.csv)
or [Relative Link](../../data/sample/df_listings_v11_sample.csv)

## Model Training

I test on a variety of regression models, and use cross-validation via RandomGridSearchCV to optimize the
hyperparameters while ensuring my model does not suffer from overfitting.

My main focus was xgboost since it was the most consistently well performing model. I also included neural networks when scaling my data, to see if I could create a deep learning model capable of outperforming my best performing non-neural network. 

I also experimented with using autoencoders and pca to perform dimensionality reduction, in the hope that it would improve the overall performance of the model. However the results were systemically better when using the data without dimensionality reduction using either autoencoding or Principal Component Analysis.
<br>[all_models_except_neural_networks_with_pca.py](all_models_except_neural_networks_with_pca.py)
<br>[neural_networks_model_with_pca.py](neural_networks_model_with_pca.py)

## Results

The final results of my trained models are available at [Markdown Version](../../process/F_evaluate_model/markdown/Summary.md) and  [HTML Version](../../process/F_evaluate_model/html/Summary.html)

The best performing model was XG Boost, when using the highest number of features.

[xg_boost__tree___v11_.html](..%2F..%2Fprocess%2FF_evaluate_model%2Fhtml%2Fxg_boost__tree___v11_.html)<br>
[xg boost (tree) (v11).md](..%2F..%2Fprocess%2FF_evaluate_model%2Fmarkdown%2Fxg%20boost%20%28tree%29%20%28v11%29.md)<br>