# ML project

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
    - source2: [target_encoding/capstoneProject__xgboost_target_encoding.ipynb](target_encoding/capstoneProject__xgboost_target_encoding.ipynb)
    - notice this notebook is linked to the data_analysis notebook
    - tried different xgboost properties: max_bepth, eta
    - found best xgboost arguments with smallest depth:  `max_depth: 20, eta: 0.6`
    - Columns selected for training: 
        - `x = [newlyConst	balcony	hasKitchen	cellar	livingSpace	lift	noRooms	garden	heating	neighbourhoods]`,
        - `y = 'baseRent'` (numerical)
    - found `MAE = 219` and `Model max deviation for 50: 27.391 percent`
4. Trained a neural network Keras model:
    - source: [capstoneProject_keras.ipynb](capstoneProject_keras.ipynb)
    - notice this notebook is linked to the data_analysis notebook
    - coverted categorical values to numerical values using LoadEncoding
    - converted booleans to ints
    - used different Dense layers with various units
    - tried different Keras parameters: learning_rate, batch_size, epochs, optimizer
    - found best Keras parameters:  `learning_rate: 0.01, batch_size: 50, epochs: 40`
    - Columns selected for training: 
        - `x = [newlyConst	balcony	hasKitchen	cellar	livingSpace	lift	noRooms	garden	heating	neighbourhood]`,
        - `y = 'baseRent'` (numerical)
    - found `MAE = 265`
5. Built model_training script using xgboost because it was most acurate.
    - source: [server/train_model.py](server/train_model.py)
6. Created web server to serve model using an API
    - server app source: [server/](server/)
    - server python file source: [server/serve.py](server/serve.py)
    - note the web server can do batch predictions to improuve performance. Reqeust payload accepts an array of data and will return an array of predictions in same order.
    - web server will catch some exceptions to return user friendly error messages and correct Status Code.
    - server can be started using vanila Python or Unicorn.
    - see below how to start it and how to call it
7. Created Docker image with the web server
    - source: [server/Dockerfile](server/Dockerfile)
    - docker image is serving the API on port 9000
    - see below how to build and run the docker image
8. Deployed to DigitalOcean using the docker image.
    - see below how to call ML-project1 running in cloud
    - this was deployed manually due to lack of time to do proper CI

# Steps to run the application.

##
Unzip `immo_data.csv.zip`, file is too big for Github.

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

(base) ➜  server git:(main) docker build -t capstone_project:v0.1 . 
Sending build context to Docker daemon  1.674MB
Step 1/8 : FROM python:3.8.12-slim
 ---> 32a5625aad35
Step 2/8 : RUN pip install pipenv
 ---> Using cache
 ---> 262147d37546
Step 3/8 : WORKDIR /app
 ---> Using cache
 ---> 153736e2bb7e
Step 4/8 : COPY ["Pipfile", "Pipfile.lock", "./"]
 ---> 9789f4690cef
Step 5/8 : RUN pipenv install --system --deploy
 ---> Running in c97d59df307c
Installing dependencies from Pipfile.lock (73e6b9)...
Removing intermediate container c97d59df307c
 ---> a0da06768ded
Step 6/8 : COPY ["serve.py", "model_xg_0.6_20.bin", "./"]
 ---> 776a41c6989a
Step 7/8 : EXPOSE 9696
 ---> Running in 9ee1152f0d87
Removing intermediate container 9ee1152f0d87
 ---> cbbecfa9784d
Step 8/8 : ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "serve:app"]
 ---> Running in 7e5fdaa73af3
Removing intermediate container 7e5fdaa73af3
 ---> 3b7063c50e92
Successfully built 3b7063c50e92
Successfully tagged capstone_project:v0.1
```

## Run docker image
```bash
(base) ➜  server git:(main) ✗ docker run -ti --rm -p 9000:9696 capstone_project:v0.1
[2021-12-12 17:36:42 +0000] [1] [INFO] Starting gunicorn 20.1.0
[2021-12-12 17:36:42 +0000] [1] [INFO] Listening at: http://0.0.0.0:9696 (1)
[2021-12-12 17:36:42 +0000] [1] [INFO] Using worker: sync
[2021-12-12 17:36:42 +0000] [7] [INFO] Booting worker with pid: 7
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
*   Trying 127.0.0.1:9000...
* Connected to localhost (127.0.0.1) port 9000 (#0)
> POST /predict HTTP/1.1
> Host: localhost:9000
> User-Agent: curl/7.71.1
> Accept: */*
> Content-Type: application/json
> Content-Length: 530
> 
* upload completely sent off: 530 out of 530 bytes
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Server: gunicorn
< Date: Sun, 12 Dec 2021 17:37:45 GMT
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

## Run docker image from my docker hub repository:
- public Docker image: https://hub.docker.com/repository/docker/razorcd/capstone_project/general
```bash
docker run -ti --rm -p 80:9696 razorcd/capstone_project:v0.1
```


## Access ML project deployed in DigitaOcean Cloud
```bash
(base) ➜  ~ curl -v -X POST http://206.189.61.226/predict \
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
*   Trying 206.189.61.226:80...
* Connected to 206.189.61.226 (206.189.61.226) port 80 (#0)
> POST /predict HTTP/1.1
> Host: 206.189.61.226
> User-Agent: curl/7.71.1
> Accept: */*
> Content-Type: application/json
> Content-Length: 884
> 
* upload completely sent off: 884 out of 884 bytes
* Mark bundle as not supporting multiuse
< HTTP/1.1 200 OK
< Server: gunicorn
< Date: Sun, 12 Dec 2021 18:24:56 GMT
< Connection: close
< Content-Type: application/json
< Content-Length: 52
< 
{"prediction":[{"baseRent":907},{"baseRent":1040}]}
```


## TODO ckecklist:

 - [x] find Dataset
 - [x] cleanup data
 - [x] perform EDA (exploratory data analysis)
 - [x] prepare data for model training
 - [x] train with linear logistic regresion
 - [x] train with xgboost
 - [x] train with Keras/Tensorflow (optional)
 - [x] create server and dockerize
 - [x] deploy to cloud (optional)
