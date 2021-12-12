# Capstone project for ML Zoomcamp

Once you move to Berlin, the hardest thing you face is renting an apartment. The competition is very high and prieces vary a lot.
I have built this ML application to predict the baseRent of an apartment based on features like living space, rooms, area, etc.
This way you can check what you can afford or if you already found an apartment you can check what would be the correct price to pay.

The dataset is based on data from 2018 - 2019. I sugest to add 10% more to the final prediction to reflect 2021 rental prices.

## Data Set

https://www.kaggle.com/corrieaar/apartment-rental-offers-in-germany

Dataset contains information from entire germany. 
I have selected only data from Berlin. This gives around 10000 records.

## Development System
  - OS: x64 Linux Ubuntu

# Project progress

Jupiter notebook has progress comments on each stept.

1. PrepareData: 
    - source: [data_analysis.ipynb](data_analysis.ipynb)
    - selected only Berlin data
    - checked and removed invalid data
    - select features
    - split data 60/20/20
    
2. Trained a Linear Regretion model:
    - source: [capstoneProject_linearRegresion.ipynb](capstoneProject_linearRegresion.ipynb)
    - notice this notebook is linked to the data_analysis notebook
    - tried different columns and countries to find most accurate combination.
    - Columns selected for training: 
        - `x = [cellar	baseRent	livingSpace	noRooms	heating	neighbourhoods]`,
        - `y = 'baseRent'` (numerical)
    - found `MAE = 257.2` and `Model max deviation for 50: 15.714 percent`
3. Trained a xgboost model:
    - source: [capstoneProject__xgboost.ipynb](capstoneProject__xgboost.ipynb)
    - notice this notebook is linked to the data_analysis notebook
    - tried different xgboost properties: max_bepth, eta
    - found best xgboost arguments with smallest depth:  `max_depth: 30, eta: 0.6`
    - Columns selected for training: 
        - `x = [newlyConst	balcony	hasKitchen	cellar	baseRent	livingSpace	lift	noRooms	garden	heating	neighbourhoods]`,
        - `y = 'baseRent'` (numerical)
    - found `MAE = 223.181` and `Model max deviation for 50: 27.198 percent`
4. Built model_training script using xgboost because it was most acurate.
    - source: [server/train_model.py](server/train_model.py)



# Steps to run the application.

## Build final model
Follow commands:

```bash
(base) ➜  project1 git:(main) ✗ cd server 

(base) ➜  server git:(main) ✗ pipenv install
Installing dependencies from Pipfile.lock (0baaa4)...
  🎃   ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉ 0/0 — 00:00:00

(base) ➜  server git:(main) ✗ python train_model.py
Data file loaded. records count: 10406
doing validation with eta=0.6, max_depth=30
mae: 232.374
mae: 236.741
mae: 229.765
mae: 252.842
mae: 243.276
mae: 230.413
mae: 243.122
mae: 230.230
mae: 233.807
mae: 232.513
validation mean mae=236.000, +-7.183
training the final model. records count= 8324
test mae=231.67433588150712
the model is saved to model_xg_0.6_30.bin

(base) ➜  server git:(main) ✗ ls
model_xg_0.6_30.bin  Pipfile  Pipfile.lock  train_model.py

```

## Start python server using Python
```bash
(base) ➜  server git:(main) ✗ python serve.py
 * Serving Flask app "low_income" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
 * Running on http://0.0.0.0:9696/ (Press CTRL+C to quit)
 * Restarting with inotify reloader
 * Debugger is active!
 * Debugger PIN: 381-893-622
```

### Start python server using Unicorn
```bash
(base) ➜  server git:(main) ✗ gunicorn --bind 0.0.0.0:9696 serve:app               
[2021-10-31 11:25:28 +0100] [28920] [INFO] Starting gunicorn 20.1.0
[2021-10-31 11:25:28 +0100] [28920] [INFO] Listening at: http://0.0.0.0:9696 (28920)
[2021-10-31 11:25:28 +0100] [28920] [INFO] Using worker: sync
[2021-10-31 11:25:28 +0100] [28922] [INFO] Booting worker with pid: 28922
```

## TODO ckecklist:

 - [x] find Dataset
 - [x] cleanup data
 - [x] perform EDA (exploratory data analysis)
 - [x] prepare data for model training
 - [x] train with linear logistic regresion
 - [x] train with xgboost
 - [ ] prepare data with Keras and train with Tensorflow
 - [ ] create server and dockerize
 - [ ] deploy to cloud (optional)