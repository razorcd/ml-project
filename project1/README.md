# Milestone 1 project from ML Zoomcamp

## Data Set

Census Income Data Set

Scope: Predict whether income exceeds $50K/yr based on goverment census data. Also known as "Adult" dataset.

Source and details: https://archive.ics.uci.edu/ml/datasets/Census+Income

## Project

Jupiter notebook has comments on each stept to:
- check and cleanup the data
- split the data (60/20/20)
- prepare hot encoding

https://github.com/razorcd/ml-training/blob/main/project1/project1.ipynb


## Project results

1. PrepareData:
    - checked and prepared data
    - split data 60/20/20
    - select features
2. Trained a Logistic Regretion model:
    - tried different columns and countries to find most accurate combination.
    - found best AUC for LogisticRegresion: `0.8800928825718033`
3. Trained a Decision Tree Classifier model:
    - used same features as logistic regretion
    - tried different Decision Tree Classifier arguments: max_depth, min_samples_leaf to find most accurate model.
    - found best Decision Tree Classifier arguments with smallest depth:  
      `auc: 0.874, depth:   10, min_samples_leaf:  100`
    - found best AUC for Decision Tree Classifier: `0.8739460370994941`
4. Trained a xgboost model:
    - used same features as logistic regretion
    - tried different xgboost properties: max_bepth, eta
    - found best xgboost arguments with smallest depth:  
      `auc: 0.883, depth:    4, eta:  0.4`
    - found best AUC for xgboost: `0.8834040882607493`
      


##TODO
[x] try linear logistic regresion
[x] try decision trees
[x] try xgboost
[ ] create server and dockerize
[ ] deploy