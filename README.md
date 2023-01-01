Details Here!


### Running the capstone submissions:
##### Run 07 submission
cd "/notebooks/capstone/step 07"
python 07_all_models_except_neural_networks.py


### Running the python model trainer on cloud:

##### Run any model except neural network
cd /notebooks/process/E_train_model/initial_models
python all_models_except_neural_networks.py

##### Run any model except neural network, with PCA for dimension reduction
cd /notebooks/process/E_train_model/pca_and_autoencoding
python all_models_except_neural_networks.py




##### Run multiple models
cd /notebooks/process/E_train_model/all_models_and_versions
python run_multiple_all_model_except_neural.py


##### Run neural network model

cd /notebooks/process/E_train_model/neural_networks/
python neural_networks_model.py


##### Run multiple neural network models

cd /notebooks/process/E_train_model/all_models_and_versions
python run_multiple_neural_networks_model.py 

-----

Old version:

Running the python model trainer on cloud:

cd /notebooks/process/E_train_model/iteration10

python it10_all_models_20221203.py 

or

python it10_ann_neural_model__20221203.py


apt-get update
apt-get install graphviz*