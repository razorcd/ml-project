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
    - found `MAE = 257.2` and `Model max deviation 50.00: 15.714 percent`

## TODO ckecklist:

 - [x] find Dataset
 - [x] cleanup data
 - [x] perform EDA (exploratory data analysis)
 - [x] prepare data for model training
 - [x] train with linear logistic regresion
 - [ ] train with decision trees
 - [ ] train with xgboost
 - [ ] prepare data with Keras and train with Tensorflow
 - [ ] create server and dockerize
 - [ ] deploy to cloud (optional)