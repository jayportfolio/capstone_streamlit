# Results from Linear Regression (Ridge)
### Summary
0.45689955499425794 is new best score (it's better than 0.456900104350899)
### Summary
not updated saved model, the previous run was better
### None
0.45689955499425794 is worse than or equal to '0.456900104350899
### Tuned Models ranked by performance, with parameter details
|   rank_test_score |   mean_test_score |   mean_fit_time |   mean_score_time |   param_model__tol | param_model__solver   |   param_model__random_state | param_model__positive   |   param_model__max_iter | param_model__fit_intercept   | param_model__copy_X   |   param_model__alpha | params2                                     |
|------------------:|------------------:|----------------:|------------------:|-------------------:|:----------------------|----------------------------:|:------------------------|------------------------:|:-----------------------------|:----------------------|---------------------:|:--------------------------------------------|
|                 1 |          0.444667 |       0.15098   |        0.00201742 |             0.001  | sag                   |                         101 | False                   |                  100000 | True                         | True                  |                1     | 0.001/sag/101/False/100000/True/True/1      |
|                 2 |          0.444667 |       0.177596  |        0.00231099 |             0.001  | sag                   |                         101 | False                   |                 1000000 | True                         | True                  |                0.1   | 0.001/sag/101/False/1000000/True/True/0.1   |
|                 3 |          0.444667 |       0.16388   |        0.00865769 |             0.001  | sag                   |                         101 | False                   |                    1000 | True                         | True                  |                1e-05 | 0.001/sag/101/False/1000/True/True/1e-05    |
|                 3 |          0.444667 |       0.185611  |        0.00373284 |             0.001  | sag                   |                         101 | False                   |                  100000 | True                         | False                 |                1e-05 | 0.001/sag/101/False/100000/True/False/1e-05 |
|                 5 |          0.444666 |       0.0156168 |        0.0018963  |             0.0001 | lsqr                  |                         101 | False                   |                    1000 | True                         | False                 |                0.1   | 0.0001/lsqr/101/False/1000/True/False/0.1   |
|                 6 |          0.444664 |       0.0813654 |        0.00210802 |             0.0001 | saga                  |                         101 | False                   |                  100000 | True                         | False                 |                1     | 0.0001/saga/101/False/100000/True/False/1   |
|                 6 |          0.444664 |       0.0856185 |        0.00241439 |             0.0001 | saga                  |                         101 | False                   |                 1000000 | True                         | True                  |                1     | 0.0001/saga/101/False/1000000/True/True/1   |
### Best and worst models obtained by tuning
![detail](./artifacts/linear_regression_(ridge)_(v06)_best_and_worst.png)
### Comparing model predictions to actual property values
![detail](./artifacts/linear_regression_(ridge)_(v06)_best_model_correlation.png)
