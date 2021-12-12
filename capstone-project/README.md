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
    - found `MAE = 257.0` and `Model max deviation for 50: 15.041 percent`
3. Trained a xgboost model:
    - source: [capstoneProject__xgboost.ipynb](capstoneProject__xgboost.ipynb)
    - notice this notebook is linked to the data_analysis notebook
    - tried different xgboost properties: max_bepth, eta
    - found best xgboost arguments with smallest depth:  `max_depth: 20, eta: 0.6`
    - Columns selected for training: 
        - `x = [newlyConst	balcony	hasKitchen	cellar	baseRent	livingSpace	lift	noRooms	garden	heating	neighbourhoods]`,
        - `y = 'baseRent'` (numerical)
    - found `MAE = 219` and `Model max deviation for 50: 27.391 percent`
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
doing validation with eta=0.6, max_depth=20
mae: 239.929
mae: 244.092
mae: 244.963
mae: 246.842
mae: 241.170
mae: 233.049
mae: 247.643
mae: 237.237
mae: 229.175
mae: 240.148
validation mean mae=240.000, +-5.459
training the final model. records count= 8324
test mae=245.7104721089857
the model is saved to model_xg_0.6_20.bin

(base) ➜  server git:(main) ✗ ls
model_xg_0.6_20.bin  Pipfile  Pipfile.lock  train_model.py
```

## Start python server using Python
```bash
(base) ➜  server git:(main) ✗ python serve.py
 * Serving Flask app "Berlin rent" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: on
 * Running on http://0.0.0.0:9696/ (Press CTRL+C to quit)
 * Restarting with inotify reloader
 * Debugger is active!
 * Debugger PIN: 201-502-766
```

### Start python server using Unicorn
```bash
(base) ➜  server git:(main) ✗ gunicorn --bind 0.0.0.0:9696 serve:app               
[2021-12-12 18:24:27 +0100] [7726] [INFO] Starting gunicorn 20.1.0
[2021-12-12 18:24:27 +0100] [7726] [INFO] Listening at: http://0.0.0.0:9696 (7726)
[2021-12-12 18:24:27 +0100] [7726] [INFO] Using worker: sync
[2021-12-12 18:24:27 +0100] [7728] [INFO] Booting worker with pid: 7728
```

## Request prediction using server API:
```bash
(base) ➜  ~ curl -v -X POST \
  http://localhost:9696/predict \
  -H 'Content-Type: application/json' \
  -d '{
        "data":[
                {
                    "neighbourhood": "friedrichshain",
                    "heating": "normal",
                    "newlyConst": false,
                    "balcony": true,
                    "hasKitchen": true,
                    "cellar": true,
                    "livingSpace": 62,
                    "lift": false,
                    "noRooms": 2,
                    "garden": false
                },
                {
                    "neighbourhood": "Steglitz",
                    "heating": "normal",
                    "newlyConst": false,
                    "balcony": false,
                    "hasKitchen": true,
                    "cellar": true,
                    "livingSpace":75,
                    "lift": false,
                    "noRooms": 3,
                    "garden": true
                }
        ]
}'
Note: Unnecessary use of -X or --request, POST is already inferred.
*   Trying 127.0.0.1:9696...
* Connected to localhost (127.0.0.1) port 9696 (#0)
> POST /predict HTTP/1.1
> Host: localhost:9696
> User-Agent: curl/7.71.1
> Accept: */*
> Content-Type: application/json
> Content-Length: 530
> 
* upload completely sent off: 530 out of 530 bytes
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Server: gunicorn
< Date: Sun, 12 Dec 2021 17:25:50 GMT
< Connection: close
< Content-Type: application/json
< Content-Length: 52
< 
{
    "prediction":[
        {
            "baseRent": 907
        },
        {
            "baseRent":1040
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