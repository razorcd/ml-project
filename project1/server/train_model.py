#!/usr/bin/env python
# coding: utf-8

import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score


# parameters

eta = 0.4
max_depth = 4
output_file = f'model_xg_{eta}_{max_depth}.bin'


# data preparation

df = pd.read_csv('../adult.data')
print("Data file loaded. records count:", len(df))
columns = [
    'age', # continuous.
    'workclass', # Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    'fnlwgt', # continuous. final weight based on Gov CPS data
    'education', # Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    'education-num', # continuous.
    'marital-status', # Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
    'occupation', # Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
    'relationship', # Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    'race', # White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    'sex', # Female, Male.
    'capital-gain', # continuous.
    'capital-loss', # continuous.
    'hours-per-week', # continuous.
    'native-country', # United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands
    'income'
]
df.columns = columns

# column types
categorical_columns = [
    'workclass', # Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    'education', # Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    'marital-status', # Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
    'occupation', # Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
    'relationship', # Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    'race', # White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    'sex', # Female, Male.
    'native-country', # United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands
    'income'
]
numerical_columns = [
    'age', # continuous.
    'fnlwgt', # continuous. final weight based on Gov CPS data
    'education-num', # continuous.
    'capital-gain', # continuous.
    'capital-loss', # continuous.
    'hours-per-week', # continuous.
]

#selected columns for training
categorical_columns_selected = [
    'workclass', # Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    'education', # Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    'marital-status', # Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse.
    'occupation', # Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces.
    'relationship', # Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    'race', # White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    'sex', # Female, Male.
#     'native-country', # United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands
]
numerical_columns_selected = [
    'age', # continuous.
#     'fnlwgt', # continuous. final weight based on Gov CPS data
    'education-num', # continuous.
#     'capital-gain', # continuous.
#     'capital-loss', # continuous.
    'hours-per-week', # continuous.
]
y_column = 'income'

#Cleanup data

#trim all categorical fields
for column in categorical_columns:
    df[column] = df[column].str.strip()

# delete rows with unknown fields
for column in categorical_columns:
    df[column].replace({"?": np.nan}, inplace=True)
df_clean = df.dropna()    
#reset index
df_clean = df_clean.reset_index(drop=True)

# prepare y
df_clean['low_income'] = df_clean.income.str.strip() == '<=50K'
del df_clean["income"]

#Filtering only USA data
df_clean_filtered = df_clean.copy()
df_clean_filtered = df_clean_filtered[df_clean_filtered['native-country'] == 'United-States']
df_clean_filtered = df_clean_filtered[categorical_columns_selected+numerical_columns_selected+['low_income']]
print("Columns selected for training:", df_clean_filtered.columns)

#Split data
from sklearn.model_selection import train_test_split
def split_data(data, split1, split2):
    # print("df_clean_filtered length: ", len(df_clean_filtered))
    # print()

    df_full_train, df_test = train_test_split(data, test_size=0.2)
    df_train,  df_val = train_test_split(df_full_train, test_size=0.25)

    df_full_train = df_full_train.reset_index(drop=True)
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_full_train = (df_full_train.low_income == True).astype('int').values
    y_train = (df_train.low_income == True).astype('int').values
    y_val = (df_val.low_income == True).astype('int').values
    y_test = (df_test.low_income == True).astype('int').values

    del df_full_train["low_income"]
    del df_train["low_income"]
    del df_val["low_income"]
    del df_test["low_income"]

    return df_full_train, df_train, df_val, df_test, y_full_train, y_train, y_val, y_test
    # print()
    # print("df_full_train length: ", len(df_full_train))
    # print("df_train length: ", len(df_train))
    # print("df_val length: ", len(df_val))
    # print("df_test length: ", len(df_test))
    # print()
    # print("y_full_train length: ", len(y_full_train))
    # print("y_train length: ", len(y_train))
    # print("y_val length: ", len(y_val))
    # print("y_test length: ", len(y_test))


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
        'objective': 'binary:logistic',
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
    df_full_train, df_train, df_val, df_test, y_full_train, y_train, y_val, y_test = split_data(df_clean_filtered, split1=0.2, split2=0.25)
    dv, model = train(df_train, y_train, max_depth=max_depth, eta=eta)
    y_pred_val, X_val = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred_val)
    scores.append(auc)

print('validation mean auc=%.3f, +-%.3f' % (np.mean(scores), np.std(scores)))


# training the final model

print('training the final model. records count=', len(df_full_train))

df_full_train, df_train, df_val, df_test, y_full_train, y_train, y_val, y_test = split_data(df_clean_filtered, split1=0.2, split2=0.25)

final_dv, final_model = train(df_full_train, y_full_train, max_depth=max_depth, eta=eta)
y_pred_test, X_test = predict(df_test, final_dv, final_model)

auc_test = roc_auc_score(y_test, y_pred_test)

print(f'test auc={auc_test}')


# # Save the model
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')
