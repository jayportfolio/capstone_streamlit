# Results from KNN
### Summary
0.7151181271345757 is new best score (it's better than 0.7190492600348584)
### Summary
not updated saved model, the previous run was better
### None
0.7151181271345757 is worse than or equal to '0.7190492600348584
### Tuned Models ranked by performance, with parameter details
|   rank_test_score |   mean_test_score |   mean_fit_time |   mean_score_time | param_model__weights   |   param_model__p |   param_model__n_neighbors |   param_model__n_jobs | param_model__metric   |   param_model__leaf_size | param_model__algorithm   | params2                               |
|------------------:|------------------:|----------------:|------------------:|:-----------------------|-----------------:|---------------------------:|----------------------:|:----------------------|-------------------------:|:-------------------------|:--------------------------------------|
|                 1 |          0.687686 |       0.153479  |           5.5123  | distance               |                1 |                          9 |                     2 | minkowski             |                        3 | auto                     | distance/1/9/2/minkowski/3/auto       |
|                 2 |          0.687677 |       0.110104  |           4.80826 | distance               |                1 |                          9 |                     2 | minkowski             |                       90 | kd_tree                  | distance/1/9/2/minkowski/90/kd_tree   |
|                 3 |          0.687673 |       0.0333918 |           4.6101  | distance               |                1 |                          9 |                     2 | minkowski             |                      300 | kd_tree                  | distance/1/9/2/minkowski/300/kd_tree  |
|                 4 |          0.678085 |       0.0677469 |           2.97734 | distance               |                2 |                          9 |                     2 | minkowski             |                        3 | auto                     | distance/2/9/2/minkowski/3/auto       |
|                 4 |          0.678085 |       0.0952417 |           3.30375 | distance               |                2 |                          9 |                     2 | minkowski             |                        3 | kd_tree                  | distance/2/9/2/minkowski/3/kd_tree    |
|                 6 |          0.678081 |       0.0739612 |          10.4178  | distance               |                2 |                          9 |                     2 | minkowski             |                       60 | ball_tree                | distance/2/9/2/minkowski/60/ball_tree |
|                 6 |          0.678081 |       0.0552374 |           8.56924 | distance               |                2 |                          9 |                     2 | euclidean             |                       60 | ball_tree                | distance/2/9/2/euclidean/60/ball_tree |
### Best and worst models obtained by tuning
![detail](./artifacts/knn_(v06)_best_and_worst.png)
### Comparing model predictions to actual property values
![detail](./artifacts/knn_(v06)_best_model_correlation.png)
