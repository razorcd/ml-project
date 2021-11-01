# Milestone 1 project from ML Zoomcamp

## Data Set

Census Income Data Set

Scope: Predict whether income exceeds $50K/yr based on goverment census data. Also known as "Adult" dataset.

Source and details: https://archive.ics.uci.edu/ml/datasets/Census+Income

## Development System
  - OS: x64 Linux Ubuntu

# Project progress

Jupiter notebook has progress comments on each stept.

1. PrepareData: 
    - source: [project1_prepareData.ipynb](project1_prepareData.ipynb)
    - checked and prepared data
    - split data 60/20/20
    - select features
    - decided to use only USA data, other counties did not have sufficient data and salaries are varying a lot between countries.
    - Columns selected for training: 
      - `x = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'age', 'education-num', 'hours-per-week']`,
      - `y = 'low_income'` (binary verion of 'income')
2. Trained a Logistic Regretion model:
    - source: [project1_logisticRegresion.ipynb](project1_logisticRegresion.ipynb)
    - notice this notebook is linked to the PrepareData notebook
    - tried different columns and countries to find most accurate combination.
    - found best AUC for LogisticRegresion: `0.8800928825718033`
3. Trained a Decision Tree Classifier model:
    - source: [project1_decisionTree.ipynb](project1_decisionTree.ipynb)
    - notice this notebook is linked to the PrepareData notebook. It is using same features as logistic regretion.
    - tried different Decision Tree Classifier arguments: max_depth, min_samples_leaf to find most accurate model.
    - found best Decision Tree Classifier arguments with smallest depth:  
      `auc: 0.874, depth:   10, min_samples_leaf:  100`
    - found best AUC for Decision Tree Classifier: `0.8739460370994941`
4. Trained a xgboost model:
    - source: [project1_xgboost.ipynb](project1_xgboost.ipynb)
    - notice this notebook is linked to the PrepareData notebook. It is using same features as logistic regretion.
    - tried different xgboost properties: max_bepth, eta
    - found best xgboost arguments with smallest depth:  
      `auc: 0.883, depth:    4, eta:  0.4`
    - found best AUC for xgboost: `0.8834040882607493`
5. Tested AUC with different split % for full_train model with xgboost.
    - source: [project1_xgboost_bestSplit.ipynb](project1_xgboost_bestSplit.ipynb)
    - AUC was very random on different splits and decided to use the default 60/20/20 split.
6. Built model_training script using xgboost because it was most acurate.
    - source: [server/train_model.py](server/train_model.py)
7. Created web server to serve model using an API
    - server app source: [server/](server/)
    - server python file source: [server/serve.py](server/serve.py)
    - note the web server can do batch predictions to improuve performance. Reqeust payload accepts an array of data and will return an array of predictions in same order.
    - web server will catch some exceptions to return user friendly error messages and correct Status Code.
    - server can be started using vanila Python or Unicorn.
    - see below how to start it and how to call it
8. Created Docker image with the web server
    - source: [server/Dockerfile](server/Dockerfile)
    - docker image is serving the API on port 9000
    - see below how to build and run the docker image
9. Deployed to DigitalOcean using the docker image.
    - see below how to call ML-project1 running in cloud


# Steps to run the application.

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

## Build docker image
```bash
(base) ➜  project1 git:(main) ✗ cd server 

(base) ➜  server git:(main) ✗ pipenv install
Pipfile.lock not found, creating...
Locking [dev-packages] dependencies...
Locking [packages] dependencies...
Building requirements...
Resolving dependencies...
✔ Success! 
Updated Pipfile.lock (73e6b9)!
Installing dependencies from Pipfile.lock (73e6b9)...
  🎃   ▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉ 3/3 — 00:00:00

(base) ➜  server git:(main) ✗ docker build -t project1_v0.2 .
Sending build context to Docker daemon  72.19kB
Step 1/8 : FROM python:3.8.12-slim
 ---> 32a5625aad35
Step 2/8 : RUN pip install pipenv
 ---> Using cache
 ---> 262147d37546
Step 3/8 : WORKDIR /app
 ---> Using cache
 ---> 153736e2bb7e
Step 4/8 : COPY ["Pipfile", "Pipfile.lock", "./"]
 ---> 4b7c708c6c0e
Step 5/8 : RUN pipenv install --system --deploy
 ---> Running in 8b27a6bce3b6
Installing dependencies from Pipfile.lock (73e6b9)...
Removing intermediate container 8b27a6bce3b6
 ---> 12019fd9ece2
Step 6/8 : COPY ["serve.py", "model_xg_0.4_4.bin", "./"]
 ---> 6440b0c37070
Step 7/8 : EXPOSE 9696
 ---> Running in b05be9638f9a
Removing intermediate container b05be9638f9a
 ---> 03f8796b36a5
Step 8/8 : ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "serve:app"]
 ---> Running in 591ec60d2660
Removing intermediate container 591ec60d2660
 ---> 14f83c080c2c
Successfully built 14f83c080c2c
Successfully tagged project1_v0.2:latest
```

## Run docker image
```bash
(base) ➜  server git:(main) ✗ docker run -ti --rm -p 9000:9696 project1_v0.2
[2021-10-31 10:41:59 +0000] [1] [INFO] Starting gunicorn 20.1.0
[2021-10-31 10:41:59 +0000] [1] [INFO] Listening at: http://0.0.0.0:9696 (1)
[2021-10-31 10:41:59 +0000] [1] [INFO] Using worker: sync
[2021-10-31 10:41:59 +0000] [7] [INFO] Booting worker with pid: 7
```
! notice Docker server API is exposed on port 9000

## Call API on Dockerized server:
```bash
(base) ➜  ~ curl -v -X POST \                                                 
  http://localhost:9000/predict \
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
*   Trying 127.0.0.1:9000...
* Connected to localhost (127.0.0.1) port 9000 (#0)
> POST /predict HTTP/1.1
> Host: localhost:9000
> User-Agent: curl/7.71.1
> Accept: */*
> Content-Type: application/json
> Content-Length: 579
> 
* upload completely sent off: 579 out of 579 bytes
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Server: gunicorn
< Date: Sun, 31 Oct 2021 10:46:55 GMT
< Connection: close
< Content-Type: application/json
< Content-Length: 123
< 
{"predictions":[{"low_income":true,"low_income_probability":"0.94"},{"low_income":false,"low_income_probability":"0.28"}]}
```

## Run docker image from my docker hub repository:
- public Docker image: https://hub.docker.com/r/razorcd/ml-project1_v0.2
```bash
docker run -ti --rm -p 9000:9696 razorcd/ml-project1_v0.2
```

## Access ML project deployed in DigitaOcean Cloud
```bash
(base) ➜  ~ curl -v -X POST http://206.189.61.226/predict \
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
*   Trying 206.189.61.226:80...
* Connected to 206.189.61.226 (206.189.61.226) port 80 (#0)
> POST /predict HTTP/1.1
> Host: 206.189.61.226
> User-Agent: curl/7.71.1
> Accept: */*
> Content-Type: application/json
> Content-Length: 1069
> 
* upload completely sent off: 1069 out of 1069 bytes
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Server: gunicorn
< Date: Mon, 01 Nov 2021 00:02:11 GMT
< Connection: close
< Content-Type: application/json
< Content-Length: 123
< 
{"predictions":[{"low_income":true,"low_income_probability":"0.96"},{"low_income":false,"low_income_probability":"0.34"}]}
```

## TODO
  - [x] prepare data
  - [x] train with linear logistic regresion
  - [x] train with decision trees
  - [x] train with xgboost
  - [x] create server and dockerize
  - [x] deploy to cloud (optional)