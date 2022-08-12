# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### NAME HERE

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
After looking at the predictions I noticed some of the predictions of count had a negative values and the kaggle submission wouldnt not accpet them 
,The negative predicitions needed to be set to 0 for the submission to accpet it.

### What was the top ranked model that performed?
WeightedEnsemble_L3 

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
There were no missing values It was mostly missclassifcation of the categorical features and the datetime that were set to int type .
The additional features were added by separating the datetime into hour, day, and month .
The "season" , "weather" columns type changed to category.

### How much better did your model preform after adding additional features and why do you think that is?
After adding the new features the model performed almost 1.5 times better than the initial model and that is correctly classifying the category columns type 
and seprating the datetime into hours, months was of importance to the accuarcy of the prediction .

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
The hpo model performed better than the new_feature model but not by much because the new_feature model had already good score .

### If you were given more time with this dataset, where do you think you would spend more time?
My hpo model got close score to the top ranking model on kaggle on the first try and the  paramters were chosen to minimize runtime , If i fine-tune the hyperparamters
and increase the training time I would get on of the top scores for this dataset on kaggle.

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
| model        | timelimit | presets      | hpo'GBM'                                                                              | hpo'NN_TORCH'                                                                                                                                                                                                    | score   |
|--------------|-----------|--------------|---------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| initial      | 600s      | best_quality | default                                                                               | default                                                                                                                                                                                                          | 1.81191 |
| add_features | 600s      | best_quality | default                                                                               | default                                                                                                                                                                                                          | 0.67912 |
| hpo_model    | 600s      | best_quality | 'num_boost_round': 100 <br>'num_leaves': ag.space.Int(lower=26, upper=66, default=36) | 'num_epochs': 10<br>'learning_rate': ag.space.Real(1e-4, 1e-2, default=5e-4, log=True)<br>'activation': ag.space.Categorical('relu', 'softrelu', 'tanh')<br>'dropout_prob': ag.space.Real(0.0, 0.5, default=0.1) | 0.48490 |

### Create a line plot showing the top model score for the three (or more) training runs during the project.


![model_train_score.png](img/model_train_score.png)

### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.


![model_test_score.png](img/model_test_score.png)

## Summary
Initially  I ran the model dataset with default values .
Then ran the data on a second model after EDA added new featrues and coorected types of the categorical features ,The score of the second model was much better than the initial one.
For the last model I did hyperparamter opimization and it produced the best scorevalue.
