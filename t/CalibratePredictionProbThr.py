# %% [markdown]
# [![CyVers](https://i.imgur.com/yyhmZET.png)](https://www.cyvers.ai/)
# 
# # Calibrate Prediction Probability Threshold
# 
# This notebooks displays the process of tuning the threshold to get the best recall give a _false alarm rate_.
# 
# > Notebook by:
# > - Royi Avital Royi@cyvers.ai
# 
# ## Revision History
# 
# | Version | Date       | User        |Content / Changes                                                   |
# |---------|------------|-------------|--------------------------------------------------------------------|
# | 0.1.000 | 05/09/2022 | Royi Avital | Matching version 0.8 of the API                                    |
# |         |            |             |                                                                    |

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


for catColName in lCatRawFeatures:
    if catColName in dfX.columns:
        dfX[catColName] = dfX[catColName].astype('category', copy = False)

# %% Train Model

xgbModel = XGBClassifier(n_estimators = 250, tree_method = 'hist', max_depth = 10,  random_state = seedNum, enable_categorical = True)
xgbModel.fit(dfX, dsY)

vY = dsY.to_numpy()

# %% Predict

mYPredProb  = xgbModel.predict_proba(dfX)
mYPredProb  = mYPredProb / np.sum(mYPredProb, axis = 1).reshape(-1, 1) #<! To have sum of 1 (We can do better with Soft Max or calibration)
vYPredProb  = mYPredProb[:, 1] #<! Probability for Label 1
vYPred      = xgbModel.predict(dfX)

# %% Reproduce vYPred from mYPredProb

vT = np.argmax(mYPredProb, axis = 1)

if np.all(vT == vYPred):
    print('The vector `vT` matches `vYPred`')
else:
    print('The vector `vT` does not match `vYPred`')

vT = np.where(vYPredProb > 0.5, 1, 0)

if np.all(vT == vYPred):
    print('The vector `vT` matches `vYPred`')
else:
    print('The vector `vT` does not match `vYPred`')


# %% Analysis of Results

DisplayConfusionMatrix(vY, vYPred, lClasses = xgbModel.classes_)

dsScoreSumm = GenClassifierSummaryResults(vY, vYPred)
dfScoreSummary  = pd.DataFrame(dsScoreSumm, columns = ['Score'])
dfScoreSummary

# %% Analysis of Prediction Probability

# Since the data is imbalanced, we need to process each group on its own.
vE = np.linspace(0, 1, 20) #<! Edges
vY0Idx = np.argwhere(vY == 0)
vY1Idx = np.argwhere(vY == 1)
vN0, *_ = np.histogram(vYPredProb[vY0Idx], bins = vE, density = True)
vN1, *_ = np.histogram(vYPredProb[vY1Idx], bins = vE, density = True)

# %%

edgeWidth = np.mean(np.diff(vE))

hF, hA = plt.subplots(figsize = (20, 10))
hA.bar(x = vE[:-1], height = vN0, width = edgeWidth, align = 'edge', label = 'Class 0')
hA.bar(x = vE[:-1], height = vN1, width = edgeWidth, align = 'edge', label = 'Class 1')
hF.legend()


# %% Optimize the Threshold
# We need Anton to first get us a real trained model
