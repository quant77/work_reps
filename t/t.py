'''
Feature statistics - absent
Visualization - absent
Preprocessing
Data cleaning - absent
Feature engineering - absent
Modeling
Model selection - SVM and decision tree
Hyperparameters tuning
Train-validation-test split / cross-validation, etc. - train/test split, no CV
Evaluation(metrics) - accuracy and confusion matrix
Feature importance - absent
'''

import pandas as pd
import numpy as np



import datetime
import os
import random
import warnings


from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Metrics
from sklearn.metrics import confusion_matrix, fbeta_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedGroupKFold, train_test_split

from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from bokeh.plotting import figure, show



df = pd.read_csv('dataset_dota_kills.csv',index_col= 0)#False)

df['score_diff'] = df['first_team_score'] - df['second_team_score']
df['rank_diff'] = df['first_team_rating'] - df['second_team_rating']


import matplotlib.pyplot as plt
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
df_ = df[np.abs(df['score_diff']).isin([30,0,1])]#df_ = df[np.abs(df['score_diff']) == 30]
ax.scatter(df['first_team_win_probability'], df['rank_diff'], df['score_diff'] ) # plotting the clusters
#ax.xlabel("prob") # X-axis label
#ax.ylabel("r_diff") # Y-axis label
#plt.zlabel("score_diff")
plt.show()