# ---- Imports

# General
import os
import pandas as pd
import numpy as np

#from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


# Category encoders
import category_encoders
from category_encoders.ordinal import OrdinalEncoder
from category_encoders.one_hot import OneHotEncoder

# Imputer
from sklearn.preprocessing import Imputer

# Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import make_pipeline

# Classifiers
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# Validation
from sklearn.model_selection import train_test_split, cross_val_score

# Metrics
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, confusion_matrix,precision_score,recall_score,f1_score

# Serialize
from sklearn.externals import joblib
import pickle
import json

#--------------------------------------------------------------#
#READ DATA
#--------------------------------------------------------------#

SEED = 42

# Read X train
path = './data/X_train.csv'
original_X = pd.read_csv(path)
index_col = original_X.index
original_X = original_X.set_index(index_col)
original_X.head()

# Read y train
path = './data/y_train.csv'
original_y = pd.read_csv(path)
original_y = original_y.rename(columns = {'0':'label'})
label_col = 'label'
original_y.head()

# Join
original_data = original_X.join(original_y,how='inner')
original_data.head()

#--------------------------------------------------------------#
#PIPELINE
#--------------------------------------------------------------#

pipeline = make_pipeline(
    category_encoders.OneHotEncoder(handle_unknown='impute'),
    Imputer(strategy='mean'),
    LogisticRegression(),
)

#--------------------------------------------------------------#
#FIT AND SERIALIZE
#--------------------------------------------------------------#

X_train = original_data.drop(columns=['label'])
y_train = original_data.label

pipeline.fit(X_train, y_train)

path = '../heroku-model-deploy-master/'

with open(path+'columns.json', 'w') as fh:
    json.dump(X_train.columns.tolist(), fh)

with open(path+'dtypes.pickle', 'wb') as fh:
    pickle.dump(X_train.dtypes, fh)

joblib.dump(pipeline, path+'pipeline.pickle')
