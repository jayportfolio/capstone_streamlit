# Results from Linear Regression (Ridge)
### Summary
0.4707818387829007 is new best score (it's better than 0.4708062463710102)
### Summary
not updated saved model, the previous run was better
### None
0.4707818387829007 is worse than or equal to '0.4708062463710102
### Tuned Models ranked by performance, with parameter details
|   rank_test_score |   mean_test_score |   mean_fit_time |   mean_score_time |   param_model__tol | param_model__solver   |   param_model__random_state | param_model__positive   |   param_model__max_iter | param_model__fit_intercept   | param_model__copy_X   |   param_model__alpha | params2                                      |
|------------------:|------------------:|----------------:|------------------:|-------------------:|:----------------------|----------------------------:|:------------------------|------------------------:|:-----------------------------|:----------------------|---------------------:|:---------------------------------------------|
|                 1 |          0.468503 |       0.273845  |        0.00915345 |             0.01   | svd                   |                         101 | False                   |                  100000 | True                         | True                  |              10      | 0.01/svd/101/False/100000/True/True/10       |
|                 2 |          0.468503 |       0.2526    |        0.00723815 |             0.01   | svd                   |                         101 | False                   |                     100 | True                         | True                  |             100      | 0.01/svd/101/False/100/True/True/100         |
|                 2 |          0.468503 |       0.216481  |        0.00729108 |             0.0001 | svd                   |                         101 | False                   |                    1000 | True                         | True                  |             100      | 0.0001/svd/101/False/1000/True/True/100      |
|                 4 |          0.468502 |       0.0701621 |        0.0123777  |             0.001  | cholesky              |                         101 | False                   |                     100 | True                         | True                  |               0.001  | 0.001/cholesky/101/False/100/True/True/0.001 |
|                 5 |          0.468502 |       0.46208   |        0.0117561  |             0.001  | svd                   |                         101 | False                   |                  100000 | True                         | False                 |               0.0001 | 0.001/svd/101/False/100000/True/False/0.0001 |
|                 6 |          0.468502 |       0.479358  |        0.00781361 |             0.0001 | saga                  |                         101 | False                   |                    1000 | True                         | False                 |              10      | 0.0001/saga/101/False/1000/True/False/10     |
|                 7 |          0.468501 |       0.463967  |        0.00607769 |             0.001  | sag                   |                         101 | False                   |                 1000000 | True                         | False                 |               1      | 0.001/sag/101/False/1000000/True/False/1     |
### Best and worst models obtained by tuning
![detail](./artifacts/linear_regression_(ridge)_(v10)_best_and_worst.png)
### Comparing model predictions to actual property values
![detail](./artifacts/linear_regression_(ridge)_(v10)_best_model_correlation.png)
