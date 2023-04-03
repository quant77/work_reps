# %% [markdown]

# # Train a Model
# This file trains a model

# %% Loading Packages

# General Tools
from cProfile import label
import numpy as np
import scipy as sp
import pandas as pd

# Misc
import datetime
import os
import pickle
from platform import python_version
import random
import warnings

# EDA Tools
import ppscore as pps #<! See https://github.com/8080labs/ppscore -> pip install git+https://github.com/8080labs/ppscore.git

# Machine Learning
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.manifold import TSNE
# from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Metrics
from sklearn.metrics import confusion_matrix, fbeta_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedGroupKFold, train_test_split

# Ensemble Engines
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# Visualization
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from bokeh.plotting import figure, show

# %% Configuration

warnings.filterwarnings("ignore")

seedNum = 512
np.random.seed(seedNum)
random.seed(seedNum)

sns.set_theme() #>! Apply SeaBorn theme

# %% Constants

DATA_FOLDER_NAME    = 'BlockChainAttacksDataSet'
DATA_FOLDER_PATTERN = 'DataSet001'
DATA_FILE_EXT       = 'csv'

PROJECT_DIR_NAME = 'CyVers' #<! Royi: Anton, don't change it, it should be a team constant
PROJECT_DIR_PATH = os.path.join(os.getcwd()[:os.getcwd().find(PROJECT_DIR_NAME)], PROJECT_DIR_NAME) #>! Pay attention, it will create issues in cases you name the folder `CyVersMe` or anything after / before `CyVers`

# Feature extractors constants

TRAIN_BY_TSX    = 1
TRAIN_BY_FILES  = 2

TIME_STAMP_FORMAT = '%Y_%m_%d_%H_%M_%S' #<! For the strftime() formatter

MODEL_FILE_NAME = 'Model'
MODEL_FILE_EXT  = 'pkl' #<! Used to be JSON for XGBoost, Needs to figure it out

# %% CyVers Packages
from DataSetsAuxFun import *

# %% Parameters

# We work according to version 0.8 API.
# See https://github.com/CyVers-AI/CyVersManagement/blob/main/AiTeamOnBoarding.md.

lRawFeatures        = ['Transaction ID', 'Block Time', 'Sender ID', 'Receiver ID', 'Receiver Type', 'Amount', 'Currency', 'Currency Hash', 'Currency Type', 'Amount [USD]', 'Label', 'Risk Level']
lProcessedFeatures  = [FeatureName.AMOUNT_MAX_ASSET.name, FeatureName.AMOUNT_MAX_USR.name, FeatureName.TIME_DIFF_MEDIAN_USR.name, FeatureName.TIME_DIFF_MEDIAN_ASSET.name]
lSelectedFeatures   = ['Receiver Type', 'Amount', 'Currency', 'Amount [USD]'] + lProcessedFeatures

dataSetRotoDir = os.path.join(PROJECT_DIR_PATH, DATA_FOLDER_NAME)

# Features Analysis
numCrossValPps = 4

# Training
trainMode = TRAIN_BY_FILES
testSetRatio = 1 / 3
numKFolds = 3
gridSearchScore = 'f1' #<! Use strings from `sklearn.metrics.get_scorer_names()`
gridSearchScore = 'recall' #<! We need to have better PD

# Amount USD Outlier threshold
amountUsdOutlierThr = 1e9

randomState = 42


# %% Loading / Generating Data
lCsvFile = ExtractCsvFiles(dataSetRotoDir, folderNamePattern = DATA_FOLDER_PATTERN)
print(f'The number of file found: {len(lCsvFile)}')

# dfData = pd.read_csv(os.path.join(DATA_FOLDER_NAME, csvFileName))
dfData, dAssetFile = LoadCsvFilesDf(lCsvFile, verifySingleSenderId = False, verifyColumns = False, baseFoldePath = '')
numRows, numCols = dfData.shape

print(f"The number of rows (Samples): {numRows}, The number of columns: {numCols}, number of unique sender id's: {dfData['Sender ID'].unique().shape}")
print(f'The data list of columns is: {dfData.columns} with {len(dfData.columns)} columns')


# %% Pre Process Data

dfData = PreProcessData(dfData, updateInplace = True, amountUsdOutlierThr = amountUsdOutlierThr)

# TODO: Define the features which are categorial

# %% Instantiate the Pandas Extension
print('Instantiate the Pandas Extension')
print(f'The number of assets in the data: {dfData.GrpBySender.numGrps}')

# %% Calculate Features

dfFeatures = ApplyListOfFeatures(dfData, lProcessedFeatures)

# %% Pre Process Data for Training

## TODO: Run a pipeline to optimize the model (By a function)
dfX = dfFeatures[lSelectedFeatures].copy()
dfX.replace([np.inf, -np.inf], np.nan, inplace = True)
dfX.fillna(0, inplace = True)
dfX.rename(columns = {'Amount [USD]': 'Amount USD'}, inplace = True)
dsY = dfData['Label']


for catColName in lCatFeatures:
    if catColName in dfX.columns:
        dfX[catColName] = dfX[catColName].astype('category', copy = False)

# %% Train Model

xgbModel = XGBClassifier(n_estimators = 250, tree_method = 'hist', max_depth = 10,  random_state = seedNum, enable_categorical = True)
xgbModel.fit(dfX, dsY)

# %% Analysis of Results

# %% Save Model

folderPostfix   = datetime.datetime.now().strftime(TIME_STAMP_FORMAT)
folderName      = MODEL_FILE_NAME + '_' + folderPostfix

modelFileName   = MODEL_FILE_NAME + '.' + MODEL_FILE_EXT

if not os.path.exists(folderName):
    os.mkdir(folderName)

# xgbModel.save_model(os.path.join(folderName, modelFileName))
pickle.dump(xgbModel, open(os.path.join(folderName, modelFileName), "wb"))
pickle.dump(lRawFeatures, open(os.path.join(folderName, 'lRawFeatures.pkl'), "wb"))
pickle.dump(lProcessedFeatures, open(os.path.join(folderName, 'lProcessedFeatures.pkl'), "wb"))
pickle.dump(lSelectedFeatures, open(os.path.join(folderName, 'lSelectedFeatures.pkl'), "wb"))
