#!/usr/bin/env python
# coding: utf-8

import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn import metrics


# parameters

eta = 0.6
max_depth = 20
output_file = f'model_xg_{eta}_{max_depth}.bin'


# data preparation

df = pd.read_csv('../immo_data_final.csv')
print("Data file loaded. records count:", len(df))

# column types
categorical_columns = [
    'neighbourhoods', # 'Spandau' 'Weißensee' 'Mitte' 'Kreuzberg' 'Tiergarten' 'Köpenick' Marzahn' 'Hohenschönhausen_Hohenschönhausen' 'Hellersdorf' 'Berg_Prenzlauer_Berg' 'Buchholz_Pankow' 'Charlottenburg' 'Tempelhof' 'Neukölln' 'Wilmersdorf' 'Wedding' 'Pankow' 'Friedrichshain' 'Reinickendorf' 'Treptow' 'Schöneberg' 'Lichtenberg' 'Steglitz' 'Zehlendorf' 'Hohenschönhausen'
    'heating', # low, normal, high
]
numerical_columns = [
    'newlyConst', 
    'balcony', 
    'hasKitchen', 
    'cellar', 
    'baseRent',
    'livingSpace', 
    'lift', 
    'noRooms', 
    'garden', 
]

y_column = 'baseRent'

#Split data
from sklearn.model_selection import train_test_split
def split_data(data, split1, split2):
    df_full_train, df_test = train_test_split(data, test_size=0.2)
    df_train,  df_val = train_test_split(df_full_train, test_size=0.25)

    df_full_train = df_full_train.reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_full_train = df_full_train.baseRent
    y_train = df_train.baseRent
    y_val = df_val.baseRent
    y_test = df_test.baseRent

    del df_full_train["baseRent"]
    del df_train["baseRent"]
    del df_val["baseRent"]
    del df_test["baseRent"]

    return df_full_train, df_train, df_val, df_test, y_full_train, y_train, y_val, y_test

#Training 

def train(dataFrame, y, max_depth, eta):
    # Hot Encoding
    dicts = dataFrame.to_dict(orient="records")
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(dicts)
    features = dv.get_feature_names()
    dtrain = xgb.DMatrix(X, label=y, feature_names=features)

    # train
    xgb_params = {
        'eta': eta,
        'max_depth': max_depth,
        'min_child_weight': 1,
        'objective': 'reg:squarederror',
        'nthread': 8,
        'seed':1,
        'verbosity':0
    }
    model = xgb.train(xgb_params, dtrain, num_boost_round=10)
    return dv, model

def predict(dataFrame, dv, model):
    dicts = dataFrame.to_dict(orient="records")
    X = dv.transform(dicts)
    features = dv.get_feature_names()
    dval = xgb.DMatrix(X, feature_names=features)
    y_pred = model.predict(dval)
    return y_pred, X 


# validation

print(f'doing validation with eta={eta}, max_depth={max_depth}')

scores = []
for i in range(0, 10):
    df_full_train, df_train, df_val, df_test, y_full_train, y_train, y_val, y_test = split_data(df, split1=0.2, split2=0.25)
    dv, model = train(df_train, y_train, max_depth=max_depth, eta=eta)
    y_pred_val, X_val = predict(df_val, dv, model)

    mae = metrics.mean_absolute_error(y_val, y_pred_val)
    print("mae: %.3f" % (mae))
    scores.append(int(mae))
    

print('validation mean mae=%.3f, +-%.3f' % (np.mean(scores), np.std(scores)))


# training the final model

print('training the final model. records count=', len(df_full_train))

df_full_train, df_train, df_val, df_test, y_full_train, y_train, y_val, y_test = split_data(df, split1=0.2, split2=0.25)

final_dv, final_model = train(df_full_train, y_full_train, max_depth=max_depth, eta=eta)
y_pred_test, X_test = predict(df_test, final_dv, final_model)

mae_test = metrics.mean_absolute_error(y_test, y_pred_test)
print(f'test mae={mae_test}')


# # Save the model
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')
