# Milestone 1 project from ML Zoomcamp

## Data Set

Census Income Data Set

Scope: Predict whether income exceeds $50K/yr based on goverment census data. Also known as "Adult" dataset.

Source and details: https://archive.ics.uci.edu/ml/datasets/Census+Income

## Development System
  - OS: x64 Linux Ubuntu

## Project progress
Jupiter notebook has progress comments on each stept.

1. PrepareData:
    - checked and prepared data
    - split data 60/20/20
    - select features
    - Columns selected for training: 
      - `x = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'age', 'education-num', 'hours-per-week']`,
      - `y = 'low_income'` (binary verion of 'income')

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
5. Tested AUC with different split % for full_train model with xgboost.
    - AUC was very random on different splits and decided to use the default 60/20/20 split.

## Build final model
Follow commands:

```
(base) ➜  project1 git:(main) ✗ cd server 

(base) ➜  server git:(main) ✗ pipenv install
Installing dependencies from Pipfile.lock (0baaa4)...
  🎃   ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉ 0/0 — 00:00:00

(base) ➜  server git:(main) ✗ python train_model.py 
Data file loaded. records count: 32560
Columns selected for training: Index(['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'age', 'education-num', 'hours-per-week', 'low_income'], dtype='object')
doing validation with eta=0.4, max_depth=4
validation mean auc=0.887, +-0.003
training the final model. records count= 22002
test auc=0.8808422986175
the model is saved to model_xg_0.4_4.bin

(base) ➜  server git:(main) ✗ ls
model_xg_0.4_4.bin  Pipfile  Pipfile.lock  train_model.py

```



## TODO
[x] try linear logistic regresion
[x] try decision trees
[x] try xgboost
[ ] create server and dockerize
[ ] deploy