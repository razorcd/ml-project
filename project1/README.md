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
6. Built model_training script using xgboost because it was most acurate.
7. Created web server to serve model using an API
    - note the web server can do batch predictions to improuve performance. Reqeust payload accepts an array of data and will return an array of predictions in same order.
    - web server will catch some exceptions to return user friendly error messages and correct Status Code.
    - server can be started using vanila Python or Unicorn.
    - see below how to start it and how to call it


## Build final model
Follow commands:

```bash
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


## Request prediction using server API:
```bash
(base) ➜  ~ curl -v -X POST \
  http://localhost:9696/predict \
  -H 'Content-Type: application/json' \
  -d '{
        "data":[
                {
                        "workclass": "Private",
                         "education": "HS-grad",
                         "marital-status": "Divorced",
                         "occupation": "Machine-op-inspct",
                         "relationship": "Not-in-family",
                         "race": "White",
                         "sex": "Female",
                         "age": 59,
                         "education-num": 9,
                         "hours-per-week": 40
                },
                {
                        "workclass": "Private",
                         "education": "Bachelors",
                         "marital-status": "Never-married",
                         "occupation": "Exec-managerial",
                         "relationship": "Not-in-family",
                         "race": "White",
                         "sex": "Male",
                         "age": 48,
                         "education-num": 20,
                         "hours-per-week": 45
                }
        ]
}'

*   Trying 127.0.0.1:9696...
* Connected to localhost (127.0.0.1) port 9696 (#0)
> POST /predict HTTP/1.1
> Host: localhost:9696
> User-Agent: curl/7.71.1
> Accept: */*
> Content-Type: application/json
> Content-Length: 579
> 
* upload completely sent off: 579 out of 579 bytes
* Mark bundle as not supporting multiuse
* HTTP 1.0, assume close after body
< HTTP/1.0 200 OK
< Content-Type: application/json
< Content-Length: 186
< Server: Werkzeug/1.0.1 Python/3.8.8
< Date: Sun, 31 Oct 2021 10:18:22 GMT
< 

{
  "predictions": [
    {
      "low_income": true, 
      "low_income_probability": "0.94"
    }, 
    {
      "low_income": false, 
      "low_income_probability": "0.28"
    }
  ]
}
```


## TODO
[x] try linear logistic regresion
[x] try decision trees
[x] try xgboost
[ ] create server and dockerize
[ ] deploy