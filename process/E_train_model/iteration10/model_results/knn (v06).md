# Results from KNN
### Summary
0.7028622217461209 is new best score (it's better than 0.7190492600348584)
### Summary
not updated saved model, the previous run was better
### None
0.7028622217461209 is worse than or equal to '0.7190492600348584
### Tuned Models ranked by performance, with parameter details
|   rank_test_score |   mean_test_score |   mean_fit_time |   mean_score_time | param_model__weights   |   param_model__p |   param_model__n_neighbors |   param_model__n_jobs | param_model__metric   |   param_model__leaf_size | param_model__algorithm   | params2                              |
|------------------:|------------------:|----------------:|------------------:|:-----------------------|-----------------:|---------------------------:|----------------------:|:----------------------|-------------------------:|:-------------------------|:-------------------------------------|
|                 1 |          0.67461  |       0.0398347 |           2.03791 | distance               |                1 |                          5 |                     2 | minkowski             |                       30 | kd_tree                  | distance/1/5/2/minkowski/30/kd_tree  |
|                 2 |          0.67382  |       0.0521699 |           1.43848 | distance               |                2 |                          7 |                     2 | minkowski             |                       30 | kd_tree                  | distance/2/7/2/minkowski/30/kd_tree  |
|                 3 |          0.665804 |       0.0604961 |           2.01902 | distance               |                1 |                          4 |                     2 | minkowski             |                        3 | kd_tree                  | distance/1/4/2/minkowski/3/kd_tree   |
|                 4 |          0.637503 |       0.0493302 |           4.23862 | uniform                |                1 |                          5 |                     2 | euclidean             |                        3 | ball_tree                | uniform/1/5/2/euclidean/3/ball_tree  |
|                 5 |          0.621798 |       0.0327361 |           7.50843 | uniform                |                1 |                          3 |                     2 | euclidean             |                       90 | ball_tree                | uniform/1/3/2/euclidean/90/ball_tree |
### Best and worst models obtained by tuning
![detail](./artifacts/knn_(v06)_best_and_worst.png)
### Comparing model predictions to actual property values
![detail](./artifacts/knn_(v06)_best_model_correlation.png)
