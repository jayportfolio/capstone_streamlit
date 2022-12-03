# Results from Linear Regression (Ridge)
### Dataset Version: 06
Date run:2022-12-03 15:24:11.305459

## Results
### Summary
not updated saved model, the previous run was better
0.456900104350899 is worse than or equal to '0.456900104350899

### Tuned Models ranked by performance, with parameter details
|   rank_test_score |   mean_test_score |   mean_fit_time |   mean_score_time |   param_model__tol | param_model__solver   |   param_model__random_state | param_model__positive   |   param_model__max_iter | param_model__fit_intercept   | param_model__copy_X   |   param_model__alpha | params2                                     |
|------------------:|------------------:|----------------:|------------------:|-------------------:|:----------------------|----------------------------:|:------------------------|------------------------:|:-----------------------------|:----------------------|---------------------:|:--------------------------------------------|
|                 1 |          0.444667 |       0.0930409 |        0.00485404 |             0.001  | sag                   |                         101 | False                   |                   10000 | True                         | False                 |                0.001 | 0.001/sag/101/False/10000/True/False/0.001  |
|                 2 |          0.444664 |       0.305608  |        0.00187794 |             1e-05  | sag                   |                         101 | False                   |                     100 | True                         | True                  |                0.001 | 1e-05/sag/101/False/100/True/True/0.001     |
|                 3 |          0.444664 |       0.0121765 |        0.00202743 |             1e-05  | cholesky              |                         101 | False                   |                     100 | True                         | True                  |               10     | 1e-05/cholesky/101/False/100/True/True/10   |
|                 4 |          0.444664 |       0.0317977 |        0.00182915 |             0.001  | svd                   |                         101 | False                   |                   10000 | True                         | False                 |               10     | 0.001/svd/101/False/10000/True/False/10     |
|                 5 |          0.444663 |       0.0160649 |        0.00165645 |             1e-05  | lsqr                  |                         101 | False                   |                   10000 | True                         | False                 |                0.001 | 1e-05/lsqr/101/False/10000/True/False/0.001 |
|                 6 |          0.444654 |       0.017688  |        0.00163062 |             1e-05  | sparse_cg             |                         101 | False                   |                     100 | True                         | True                  |              100     | 1e-05/sparse_cg/101/False/100/True/True/100 |
|                 7 |          0.444654 |       0.243023  |        0.00189082 |             0.0001 | sag                   |                         101 | False                   |                  100000 | True                         | True                  |              100     | 0.0001/sag/101/False/100000/True/True/100   |
### Best and worst models obtained by tuning
![detail](./artifacts/linear_regression_(ridge)_(v06)_best_and_worst.png)
### Best Model: Comparing model predictions to actual property values
![detail](./artifacts/linear_regression_(ridge)_(v06)_best_model_correlation.png)
## Appendix
### Data Sample
|          |   Price |   bedrooms |   bathrooms |   nearestStation |   location.latitude |   location.longitude |   latitude_deviation |   longitude_deviation | tenure.tenureType   |
|---------:|--------:|-----------:|------------:|-----------------:|--------------------:|---------------------:|---------------------:|----------------------:|:--------------------|
| 14520525 |  550000 |          3 |           1 |         0.274316 |             51.5299 |            -0.20702  |             0.03023  |              0.1026   | LEASEHOLD           |
| 27953107 |  400000 |          2 |           2 |         0.305845 |             51.5494 |            -0.4826   |             0.04967  |              0.37818  | LEASEHOLD           |
| 33593487 |  579950 |          2 |           1 |         0.438045 |             51.4472 |            -0.33877  |             0.05254  |              0.23435  | FREEHOLD            |
| 35271294 |  370000 |          2 |           1 |         0.399307 |             51.4496 |            -0.140154 |             0.050152 |              0.035734 | LEASEHOLD           |
| 44749111 |  475000 |          2 |           1 |         0.41055  |             51.37   |            -0.21241  |             0.12967  |              0.10799  | FREEHOLD            |
### Hyperparameter options for Randomized Grid Search
model__alpha = [1e-05, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]

model__fit_intercept = [True, False]

model__max_iter = [10000, 1000, 100, 100000, 1000000]

model__positive = [False]

model__copy_X = [True, False]

model__solver = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']

model__tol = [1e-05, 0.0001, 0.001, 0.01]

model__random_state = [101]

### Environment Variables
notebook_environment = local

use_gpu = False

debug_mode = False

quick_mode = False

quick_override_cv_splits = 2

quick_override_n_iter = 10

quick_override_n_jobs = 3

