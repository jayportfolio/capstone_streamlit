# Results from Random Forest
### Dataset Version: 09
Date run: 2022-12-11 19:26:15.839870

Start time: 2022-12-11 17:59:10.695847

End time: 2022-12-11 19:26:15.839864

## Results
### Summary
pickled new version of model
0.6800913841324449 is new best score (it's better than 0.2549015378334345)

### Tuned Models ranked by performance, with parameter details
|   rank_test_score |   mean_test_score |   mean_fit_time |   mean_score_time | param_model__warm_start   |   param_model__verbose |   param_model__random_state | param_model__oob_score   | param_model__n_jobs   |   param_model__n_estimators |   param_model__min_weight_fraction_leaf |   param_model__min_samples_split |   param_model__min_samples_leaf |   param_model__min_impurity_decrease | param_model__max_samples   |   param_model__max_leaf_nodes | param_model__max_features   |   param_model__max_depth | param_model__criterion   |   param_model__ccp_alpha | param_model__bootstrap   | params2                                                                          |
|------------------:|------------------:|----------------:|------------------:|:--------------------------|-----------------------:|----------------------------:|:-------------------------|:----------------------|----------------------------:|----------------------------------------:|---------------------------------:|--------------------------------:|-------------------------------------:|:---------------------------|------------------------------:|:----------------------------|-------------------------:|:-------------------------|-------------------------:|:-------------------------|:---------------------------------------------------------------------------------|
|                 1 |          0.657777 |       516.439   |         0.112312  | False                     |                      0 |                          16 | False                    |                       |                          70 |                                       0 |                               50 |                               2 |                                 5    |                            |                               | 0.5                         |                       50 | absolute_error           |                     1    | True                     | False/0/16/False/None/70/0.0/50/2/5/None/None/0.5/50/absolute_error/1/True       |
|                 2 |          0.631265 |         6.72736 |         0.0813395 | False                     |                      0 |                          16 | True                     |                       |                          70 |                                       0 |                                4 |                               2 |                                 1    |                            |                           200 |                             |                          | friedman_mse             |                     1    | True                     | False/0/16/True/None/70/0.0/4/2/1/None/200/None/None/friedman_mse/1/True         |
|                 3 |          0.623844 |       259.758   |         0.0982006 | False                     |                      0 |                         101 | True                     |                       |                          70 |                                       0 |                                4 |                               1 |                                 0    |                            |                           750 | sqrt                        |                       50 | absolute_error           |                     0.05 | True                     | False/0/101/True/None/70/0.0/4/1/0.0/None/750/sqrt/50/absolute_error/0.05/True   |
|                 4 |          0.621741 |         8.11477 |         0.0936906 | False                     |                      0 |                         101 | False                    |                       |                          70 |                                       0 |                              100 |                              50 |                                 0.25 |                            |                               | 1.0                         |                          | poisson                  |                     0.25 | True                     | False/0/101/False/None/70/0.0/100/50/0.25/None/None/1.0/None/poisson/0.25/True   |
|                 5 |          0.561713 |         5.44267 |         0.0609867 | False                     |                      0 |                           8 | False                    |                       |                          70 |                                       0 |                              200 |                              50 |                                 0.1  |                            |                            50 | 1.0                         |                          | friedman_mse             |                     5    | True                     | False/0/8/False/None/70/0.0/200/50/0.1/None/50/1.0/None/friedman_mse/5/True      |
|                 6 |          0.55934  |         5.54019 |         0.068059  | False                     |                      0 |                          15 | True                     |                       |                          70 |                                       0 |                              500 |                               2 |                                 5    |                            |                           750 | 1.0                         |                       10 | friedman_mse             |                     1    | True                     | False/0/15/True/None/70/0.0/500/2/5/None/750/1.0/10/friedman_mse/1/True          |
|                 7 |          0.543832 |         1.1565  |         0.0515256 | False                     |                      0 |                         101 | True                     |                       |                          50 |                                       0 |                              500 |                               2 |                                 0.1  |                            |                               | sqrt                        |                       50 | squared_error            |                     0.25 | True                     | False/0/101/True/None/50/0.0/500/2/0.1/None/None/sqrt/50/squared_error/0.25/True |
### Best and worst models obtained by tuning
![detail](../artifacts/random_forest__v09__best_and_worst.png)
### Best Model: Comparing model predictions to actual property values
![detail](../artifacts/random_forest__v09__best_model_correlation.png)
## Feature Importances
### Feature Importances
1. features 0 (0.216203)		bedrooms
2. features 4 (0.201366)		location.longitude
3. features 5 (0.136760)		latitude_deviation
4. features 3 (0.110894)		location.latitude
5. features 6 (0.103557)		longitude_deviation
6. features 19 (0.073799)		tenure.tenureType_LEASEHOLD
7. features 1 (0.051007)		bathrooms
8. features 2 (0.041821)		nearestStation
9. features 18 (0.041587)		tenure.tenureType_FREEHOLD
10. features 20 (0.006550)		tenure.tenureType_SHARE_OF_FREEHOLD
11. features 8 (0.003927)		feature__chain free
12. features 14 (0.002461)		feature__three bedrooms
13. features 12 (0.001717)		feature__private balcony
14. features 7 (0.001466)		feature__balcony
15. features 16 (0.001358)		feature__two double bedrooms
16. features 13 (0.001257)		feature__share of freehold
17. features 9 (0.001234)		feature__no onward chain
18. features 11 (0.001049)		feature__one bedroom
19. features 15 (0.001046)		feature__two bedrooms
20. features 10 (0.000935)		feature__off street parking
21. features 17 (0.000003)		tenure.tenureType_COMMONHOLD


### Feature Importances (Random Forest)
![detail](../artifacts/random_forest__v09__best_model_feature_importances.png)
## Comparison with other models
### Comparison with version 09 performances
|                                         |   best score |   best time |   Mean Absolute Error Accuracy |   Mean Squared Error Accuracy |   R square Accuracy |   Root Mean Squared Error | best run date              | best method                                            |
|:----------------------------------------|-------------:|------------:|-------------------------------:|------------------------------:|--------------------:|--------------------------:|:---------------------------|:-------------------------------------------------------|
| xg boost (v09)                          |     0.701117 |  119.281    |                     51922.6    |                   4.35555e+09 |            0.614431 |                65996.6    | 2022-11-30 10:16:33.388760 | random search                                          |
| catboost (v09)                          |     0.700506 |    2.82     |                     44531.4    |                   3.38321e+09 |            0.700506 |                58165.3    | 2022-11-30 13:34:39.793583 | random search(no dummies)                              |
| random forest (v09)                     |     0.680091 | 1127.32     |                     46294.6    |                   3.61382e+09 |            0.680091 |                60115      | 2022-12-11 19:26:14.332459 | random search                                          |
| knn (v09)                               |     0.644916 |    0.112408 |                     48389.1    |                   4.15737e+09 |            0.631974 |                64477.7    | 2022-11-30 12:53:42.390150 | random search                                          |
| decision tree (v09)                     |     0.619635 |    0.409451 |                     51319.1    |                   4.31524e+09 |            0.617999 |                65690.5    | 2022-12-11 14:59:07.675693 | random search                                          |
| neural network m12 mega (v09)           |     0.570335 |  556.62     |                        55.1583 |                4853.68        |            0.570335 |                   69.6683 | 2022-11-29 21:03:09.676165 | loss=4869.67 valloss=4726.34 +valsplit=0.1 stop=83/400 |
| neural network m02 two layers (v09)     |     0.540824 |  178.62     |                     60855      |                   5.67986e+09 |            0.497198 |                75364.9    | 2022-11-30 13:34:57.703544 | loss=5424.62 valloss=5263.41 +valsplit=0.1 stop=52/500 |
| neural network m01 simple (v09)         |     0.508847 |  188.63     |                     71376.9    |                   7.55056e+09 |            0.331597 |                86893.9    | 2022-11-30 13:08:10.248178 | loss=5724.92 valloss=5608.12 +valsplit=0.1 stop=42/50  |
| linear regression (ridge) (v09)         |     0.459888 |    0.350431 |                     63349.5    |                   6.10132e+09 |            0.459888 |                78110.9    | 2022-12-11 14:36:43.474759 | random search                                          |
| neural network m03 2 layers+wider (v09) |     0.4523   | 1822.49     |                     63891.8    |                   6.18704e+09 |            0.4523   |                78657.8    | 2022-12-11 18:56:27.304803 | loss=6.38e+09 valloss=6.41e+09 +valsplit=0.1 +patn=25  |
### Comparison with all model performances
|                                                          |   best score |    best time |   Mean Absolute Error Accuracy |   Mean Squared Error Accuracy |   R square Accuracy |   Root Mean Squared Error | best run date              | best method                                                                                                            |
|:---------------------------------------------------------|-------------:|-------------:|-------------------------------:|------------------------------:|--------------------:|--------------------------:|:---------------------------|:-----------------------------------------------------------------------------------------------------------------------|
| xg boost (tree) (v06)                                    |     0.727132 |  134.175     |                     41210.4    |                   3.08243e+09 |            0.727132 |                55519.6    | 2022-12-07 09:43:37.103009 | random search                                                                                                          |
| xg boost (v11) rs                                        |     0.721019 |  nan         |                     42603      |                   3.15148e+09 |            0.721019 |                56138      | nan                        | nan                                                                                                                    |
| knn (v06)                                                |     0.719049 |    0.0179159 |                     41531.1    |                   3.2181e+09  |            0.715122 |                56728.3    | 2022-11-21 18:05:21.585382 | random search                                                                                                          |
| catboost (v06)                                           |     0.715606 |   12.2565    |                     51000      |                   4.30136e+09 |            0.619227 |                65584.8    | 1999-11-13 15:26:55.706567 | random search                                                                                                          |
| xg boost (v05) rs                                        |     0.710594 |  nan         |                     42229      |                   3.21963e+09 |            0.710594 |                56741.7    | nan                        | nan                                                                                                                    |
| light gradient boosting (v06)                            |     0.706735 |   15.0439    |                     44081      |                   3.31284e+09 |            0.706735 |                57557.3    | 2022-11-16 13:59:52.612654 | random search                                                                                                          |
| xg boost (v09)                                           |     0.701117 |  119.281     |                     51922.6    |                   4.35555e+09 |            0.614431 |                65996.6    | 2022-11-30 10:16:33.388760 | random search                                                                                                          |
| catboost (v09)                                           |     0.700506 |    2.82      |                     44531.4    |                   3.38321e+09 |            0.700506 |                58165.3    | 2022-11-30 13:34:39.793583 | random search(no dummies)                                                                                              |
| catboost (v10)                                           |     0.694651 |    4.77      |                     44875      |                   3.44935e+09 |            0.694651 |                58731.2    | 2022-11-30 14:14:50.145713 | random search(no dummies)                                                                                              |
| catboost (v11)                                           |     0.689818 |   12.81      |                     45407.3    |                   3.50394e+09 |            0.689818 |                59194.1    | 2022-11-30 16:14:29.405177 | random search(no dummies)                                                                                              |
| xg boost (v06)                                           |     0.687611 |   11.4748    |                     45988.3    |                   3.52887e+09 |            0.687611 |                59404.3    | nan                        | random search                                                                                                          |
| xg boost (v10)                                           |     0.681785 |    9.30959   |                     46626.7    |                   3.59469e+09 |            0.681785 |                59955.7    | 2022-11-30 14:45:52.207314 | random search                                                                                                          |
| random forest (v09)                                      |     0.680091 | 1127.32      |                     46294.6    |                   3.61382e+09 |            0.680091 |                60115      | 2022-12-11 19:26:14.332459 | random search                                                                                                          |
| random forest - random search (vx10)                     |     0.647421 |  nan         |                     49942      |                   3.98288e+09 |            0.647421 |                63110.1    | nan                        | nan                                                                                                                    |
| knn (v09)                                                |     0.644916 |    0.112408  |                     48389.1    |                   4.15737e+09 |            0.631974 |                64477.7    | 2022-11-30 12:53:42.390150 | random search                                                                                                          |
| decision tree (v09)                                      |     0.619635 |    0.409451  |                     51319.1    |                   4.31524e+09 |            0.617999 |                65690.5    | 2022-12-11 14:59:07.675693 | random search                                                                                                          |
| decision tree (v06)                                      |     0.616727 |    0.133738  |                     59431.1    |                   5.74359e+09 |            0.491556 |                75786.5    | 2022-12-11 13:59:57.993851 | random search                                                                                                          |
| neural network m11 mega (v06)                            |     0.612318 | 2569.45      |                     56035.9    |                   4.92994e+09 |            0.563583 |                70213.6    | 2022-11-29 12:57:16.459719 | loss=2833.6 valloss=4034.41 stop=619/1000                                                                              |
| xg boost (tree) (v11)                                    |     0.603614 |   14.2104    |                     52330.4    |                   4.47774e+09 |            0.603614 |                66915.9    | 2022-11-30 20:18:59.876471 | random search                                                                                                          |
| xg boost (v04) rs                                        |     0.603522 |  nan         |                     50419.2    |                   4.50494e+09 |            0.603522 |                67118.9    | nan                        | nan                                                                                                                    |
| neural network m12 mega (v06)                            |     0.594032 |  813.27      |                     54968      |                   4.80703e+09 |            0.574463 |                69332.8    | 2022-11-29 17:08:44.480482 | loss=4386.51 valloss=4438.8 +valsplit=0.1 stop=201/400                                                                 |
| random forest (v06)                                      |     0.585876 |    1.76986   |                     60537      |                   5.63491e+09 |            0.501177 |                75066      | 2022-12-11 12:04:23.528125 | random search                                                                                                          |
| neural network m13 mega (v10)                            |     0.583716 |  142.89      |                     54668.9    |                   4.94809e+09 |            0.561977 |                70342.6    | 2022-12-01 10:27:39.663081 | loss=3878948096.0 valloss=4822886400.0 +valsplit=0.1 stop=38/400                                                       |
| xg boost (v03) rs                                        |     0.582071 |  nan         |                     51147.3    |                   4.7333e+09  |            0.574533 |                68799      | nan                        | nan                                                                                                                    |
| neural network m05 rec deep (v06)                        |     0.580348 |  604.9       |                     59357.5    |                   5.47083e+09 |            0.515701 |                73965.1    | 2022-11-29 11:41:39.682217 | loss=4908.71 valloss=4603.08 stop=214/500                                                                              |
| neural network m14 mega (v10)                            |     0.579095 | 1129.09      |                     53124      |                   4.75471e+09 |            0.579095 |                68954.4    | 2022-12-01 11:52:45.011704 | loss=4.85e+04 valloss=5.34e+04 +valsplit=0.1 stop=156/400                                                              |
| neural network m12 mega (v09)                            |     0.570335 |  556.62      |                        55.1583 |                4853.68        |            0.570335 |                   69.6683 | 2022-11-29 21:03:09.676165 | loss=4869.67 valloss=4726.34 +valsplit=0.1 stop=83/400                                                                 |
| neural network m12 mega (v10)                            |     0.567453 |  240.1       |                     55444      |                   4.88623e+06 |            0.567453 |                69901.6    | 2022-12-01 09:57:17.586487 | loss=4790.75 valloss=4998.79 +valsplit=0.1 stop=66/400                                                                 |
| decision tree - random search (vx10)                     |     0.558257 |  nan         |                     55865.4    |                   4.99011e+09 |            0.558257 |                70640.7    | nan                        | nan                                                                                                                    |
| neural network (v06)                                     |     0.556696 |  312.991     |                     66710.7    |                   6.64686e+09 |            0.411595 |                81528.3    | 2000-01-01 17:09:59.063570 | random search [input11, d^20-500-500-20-5, dense1]                                                                     |
| neural network m03 2 layers+wider (v06)                  |     0.549647 |  275.71      |                     64376.5    |                   6.26802e+09 |            0.445131 |                79170.9    | 2022-11-29 10:13:10.517896 | mse +epochs=500 +learn=0.003 +loss=5229.0478515625                                                                     |
| neural network m01 simple (v06)                          |     0.541221 |   36.2       |                     69132      |                   7.15587e+09 |            0.366536 |                84592.4    | 2022-11-29 09:13:15.856770 | recommended simple model/mse +norm +epochs=50 +learn=0.003 +endloss=5610.65771484375 +stop=17 +endloss=5511.7373046875 |
| neural network m02 two layers (v09)                      |     0.540824 |  178.62      |                     60855      |                   5.67986e+09 |            0.497198 |                75364.9    | 2022-11-30 13:34:57.703544 | loss=5424.62 valloss=5263.41 +valsplit=0.1 stop=52/500                                                                 |
| neural network simplified (v06)                          |     0.540642 |  999         |                     59373.1    |                   5.53151e+09 |            0.51033  |                74374.1    | 2022-11-20 20:03:40.645221 | recommended simple model + normalise, mse                                                                              |
| knn - random search (vx10)                               |     0.533823 |    0.0497677 |                     57566.9    |                   5.26613e+09 |            0.533823 |                72568.1    | nan                        | nan                                                                                                                    |
| neural network - random search [i64,norm,d64^6,d1] (v11) |     0.533579 |  nan         |                     57201.7    |                   5.26888e+09 |            0.533579 |                72587      | nan                        | nan                                                                                                                    |
| neural network m04 3 layers+wider (v06)                  |     0.520933 |  395.14      |                     64421.2    |                   6.26641e+09 |            0.445274 |                79160.6    | 2022-11-29 11:21:09.812732 | loss=5415.7 valloss=5095.94 stop=166/500                                                                               |
| neural network m02 two layers (v06)                      |     0.516773 |  112.54      |                     64363.1    |                   6.26209e+09 |            0.445656 |                79133.4    | 2022-11-29 09:31:18.853517 | mse +norm +epochs=50 +learn=0.003 +endloss=5785.6953125                                                                |
| neural network m01 simple (v09)                          |     0.508847 |  188.63      |                     71376.9    |                   7.55056e+09 |            0.331597 |                86893.9    | 2022-11-30 13:08:10.248178 | loss=5724.92 valloss=5608.12 +valsplit=0.1 stop=42/50                                                                  |
| knn (v10)                                                |     0.484585 |    0.29179   |                     61764.2    |                   5.82234e+09 |            0.484585 |                76304.2    | 2022-11-30 15:12:50.989371 | random search                                                                                                          |
| xg boost (v11)                                           |     0.484341 |    1.66323   |                     61556.7    |                   5.82509e+09 |            0.484341 |                76322.3    | 2022-11-30 16:55:55.436173 | random search                                                                                                          |
| xg boost (linear) (v11)                                  |     0.484341 |   12.6817    |                     62224.3    |                   5.90103e+09 |            0.477618 |                76818.2    | 2022-11-30 19:47:04.498556 | random search                                                                                                          |
| linear regression (ridge) (v10)                          |     0.470806 |    0.239057  |                     62596.2    |                   5.97799e+09 |            0.470806 |                77317.4    | 2022-12-01 19:50:08.050622 | random search                                                                                                          |
| knn (v11)                                                |     0.465113 |    0.618877  |                     62944.7    |                   6.0423e+09  |            0.465113 |                77732.2    | 2022-11-30 16:20:53.948815 | random search                                                                                                          |
| linear regression (ridge) (v09)                          |     0.459888 |    0.350431  |                     63349.5    |                   6.10132e+09 |            0.459888 |                78110.9    | 2022-12-11 14:36:43.474759 | random search                                                                                                          |
| linear regression (ridge) (v06)                          |     0.4569   |    0.28695   |                     63603.1    |                   6.13521e+09 |            0.456889 |                78327.6    | 2022-12-03 19:20:52.874336 | random search                                                                                                          |
| neural network m03 2 layers+wider (v09)                  |     0.4523   | 1822.49      |                     63891.8    |                   6.18704e+09 |            0.4523   |                78657.8    | 2022-12-11 18:56:27.304803 | loss=6.38e+09 valloss=6.41e+09 +valsplit=0.1 +patn=25                                                                  |
| knn - basic (v01)                                        |   nan        |  nan         |                     55623.7    |                   5.34585e+09 |            0.546891 |                73115.3    | nan                        | nan                                                                                                                    |
| knn - basic (v02)                                        |   nan        |  nan         |                     52181.5    |                   4.75613e+09 |            0.584356 |                68964.7    | nan                        | nan                                                                                                                    |
| knn - random search (v01)                                |   nan        |  nan         |                     52593.9    |                   4.86155e+09 |            0.587939 |                69724.8    | nan                        | nan                                                                                                                    |
| knn - random search (v02)                                |   nan        |  nan         |                     49441.2    |                   4.26278e+09 |            0.62747  |                65290      | nan                        | nan                                                                                                                    |
| knn - scaled (v01)                                       |   nan        |  nan         |                     52147.4    |                   4.86744e+09 |            0.58744  |                69767.1    | nan                        | nan                                                                                                                    |
| linear regression (ridge) - random search (v02)          |   nan        |  nan         |                     71267.9    |                   7.70239e+09 |            0.326879 |                87763.2    | nan                        | nan                                                                                                                    |
| linear regression (ridge) - random search (v03)          |   nan        |  nan         |                     70746.7    |                   7.49253e+09 |            0.326511 |                86559.4    | nan                        | nan                                                                                                                    |
| linear regression (ridge) - random search (v04)          |   nan        |  nan         |                     71834.4    |                   7.71252e+09 |            0.321224 |                87821      | nan                        | nan                                                                                                                    |
| linear regression (ridge) - random search (v05)          |   nan        |  nan         |                     63770.7    |                   6.19128e+09 |            0.443478 |                78684.7    | nan                        | nan                                                                                                                    |
| linear regression - basic (v01)                          |   nan        |  nan         |                     72921.6    |                   8.29799e+09 |            0.29667  |                91093.3    | nan                        | nan                                                                                                                    |
| xg boost (v02) rs                                        |   nan        |  nan         |                     45160.2    |                   3.62788e+09 |            0.682955 |                60231.8    | nan                        | nan                                                                                                                    |
| xg boost - basic (v02)                                   |   nan        |  nan         |                     48536.5    |                   3.97959e+09 |            0.652219 |                63084      | nan                        | nan                                                                                                                    |
## Appendix
### Data Sample
|          |   Price |   bedrooms |   bathrooms |   nearestStation |   location.latitude |   location.longitude |   latitude_deviation |   longitude_deviation | tenure.tenureType   |   feature__balcony |   feature__chain free |   feature__no onward chain |   feature__off street parking |   feature__one bedroom |   feature__private balcony |   feature__share of freehold |   feature__three bedrooms |   feature__two bedrooms |   feature__two double bedrooms |
|---------:|--------:|-----------:|------------:|-----------------:|--------------------:|---------------------:|---------------------:|----------------------:|:--------------------|-------------------:|----------------------:|---------------------------:|------------------------------:|-----------------------:|---------------------------:|-----------------------------:|--------------------------:|------------------------:|-------------------------------:|
| 14520525 |  550000 |          3 |           1 |         0.274316 |             51.5299 |            -0.20702  |             0.03023  |              0.1026   | LEASEHOLD           |                  0 |                     0 |                          0 |                             0 |                      0 |                          1 |                            0 |                         0 |                       0 |                              0 |
| 27953107 |  400000 |          2 |           2 |         0.305845 |             51.5494 |            -0.4826   |             0.04967  |              0.37818  | LEASEHOLD           |                  1 |                     0 |                          0 |                             0 |                      0 |                          0 |                            0 |                         0 |                       0 |                              1 |
| 33593487 |  579950 |          2 |           1 |         0.438045 |             51.4472 |            -0.33877  |             0.05254  |              0.23435  | FREEHOLD            |                  0 |                     0 |                          1 |                             0 |                      0 |                          0 |                            0 |                         0 |                       0 |                              0 |
| 35271294 |  370000 |          2 |           1 |         0.399307 |             51.4496 |            -0.140154 |             0.050152 |              0.035734 | LEASEHOLD           |                  1 |                     0 |                          0 |                             0 |                      0 |                          0 |                            0 |                         0 |                       0 |                              0 |
| 44749111 |  475000 |          2 |           1 |         0.41055  |             51.37   |            -0.21241  |             0.12967  |              0.10799  | FREEHOLD            |                  0 |                     0 |                          0 |                             0 |                      0 |                          0 |                            0 |                         0 |                       0 |                              0 |
### Hyperparameter options for Randomized Grid Search
model__bootstrap = [True]

model__ccp_alpha = [0.0, 0.05, 0.1, 0.25, 1, 5]

model__criterion = ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']

model__max_depth = [None, 1, 2, 5, 10, 50]

model__max_features = [None, 1.0, 'sqrt', 'log2', 0.5, 0.25, 0.1, 2]

model__max_leaf_nodes = [None, 2, 5, 10, 50, 100, 200, 500, 750]

model__max_samples = [None]

model__min_impurity_decrease = [0.0, 0.1, 0.25, 1, 5]

model__min_samples_leaf = [1, 1.5, 2, 4, 8, 50]

model__min_samples_split = [2, 4, 8, 50, 100, 200, 500]

model__min_weight_fraction_leaf = [0.0, 0.1, 0.25, 0.5]

model__n_estimators = [50, 70]

model__n_jobs = [None]

model__oob_score = [False, True]

model__random_state = [4, 8, 15, 16, 23, 42, 101]

model__verbose = [0]

model__warm_start = [False]

### Range of hyperparameter results
![detail](../artifacts/random_forest__v09__evolution_of_models_fig.png)
### Environment Variables
notebook_environment = gradient

use_gpu = True

debug_mode = False

quick_mode = False

quick_override_cv_splits = 2

quick_override_n_iter = 10

quick_override_n_jobs = 3

