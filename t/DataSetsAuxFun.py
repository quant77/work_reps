# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Machine Learning
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, precision_score, recall_score , roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split

from xgboost import XGBClassifier

# Miscellaneous
from enum import Enum, auto, unique
import functools as ft
# import random
import os
# import datetime
# from platform import python_version

import sys
import hashlib


# Typing
from typing import Dict, List

# Visualization
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from bokeh.plotting import figure, show

from DataSetsPandasExt import *
from DataSetFeaturesFun import *

#lCsvColName     = ['Transaction ID', 'Block Time', 'Transaction Time', 'Sender ID', 'Receiver ID', 'Receiver Type', 'Amount', 'Currency', 'Currency Hash', 'Currency Type', 'Amount [USD]', 'Gas Price', 'Gas Limit', 'Gas Used', 'Gas Predicted', 'Balance In', 'Balance Out', 'Label', 'Risk Level']
#lCsvColNameFlag = [True,              True,         True,               True,        True,          True,            True,     True,       True,            True,            True,           True,        True,        True,       True,            True,         True,          False,   False]  #<! Flags if a column is a must to have

# From Template
# dfTemplate      = pd.read_csv('TemplateCsv.csv')
# lCsvColName     = dfTemplate.columns.to_list()
# sCsvColName     = set(lCsvColName) #<! Set
# lCsvColNameFlag = list(dfTemplate.iloc[0, :].to_numpy().astype(np.bool8))

lCatRawFeatures    = ['Receiver Type', 'Currency', 'Currency Hash', 'Currency Type'] #<! RAW features which are categorial

@unique
class AttackType(Enum):
    # Type of data in the CSV
    SASASW = auto()
    SAMASW = auto()
    SAMAMW = auto()
    MASASW = auto()
    MAMASW = auto()
    MAMAMW = auto()

#def LoadCsvFilesDf( lCsvFileName: List, baseFoldePath = '.', verifySingleSenderId = True, verifyColumns = True, lColName = lCsvColName, lColFlag =  lCsvColNameFlag, addFileNameCol = False ) -> pd.DataFrame:
def LoadCsvFilesDf( lCsvFileName: List, lColName, lColFlag , baseFoldePath = '.', verifySingleSenderId = True, verifyColumns = True,  addFileNameCol = False ) -> pd.DataFrame:

    dAssetFile = {}

    dfData = pd.read_csv(os.path.join(baseFoldePath, lCsvFileName[0]))
    dAssetFile[dfData['Sender ID'].unique()[0]] = lCsvFileName[0]
    if addFileNameCol:
        dfData['File Name'] = os.path.basename(lCsvFileName[0])

        lColName.append('File Name')
        lColFlag.append(True)

    if verifySingleSenderId:
        vSenderId = dfData['Sender ID'].unique()
        numUniqueSenderId = len(vSenderId)
        sSenderId = ''
        for jj in range(numUniqueSenderId):
            sSenderId = f'Sender ID: {vSenderId[jj]}, '

        if numUniqueSenderId != 1:
            raise ValueError(f'The file {os.path.basename(lCsvFileName[0])} has more than a single unique `Sender ID`: {sSenderId}')

    if verifyColumns:
        lColumns = dfData.columns.to_list()
        for colName in lColumns:
            if not(colName in lColName):#if not(colName in lCsvColName):
                raise ValueError(f'The file {os.path.basename(lCsvFileName[0])} has a column name: {colName} which is not defined')
        for jj, colName in enumerate(lColName):
            if lColFlag[jj] and (not(colName in lColumns)):
                raise ValueError(f'The file {os.path.basename(lCsvFileName[0])} does not have the required column name: {colName}')

    for ii in range(1, len(lCsvFileName)):

        dfCurrData = pd.read_csv(os.path.join(baseFoldePath, lCsvFileName[ii]))
        dAssetFile[dfCurrData['Sender ID'].unique()[0]] = lCsvFileName[ii]

        if addFileNameCol:
            dfCurrData['File Name'] = os.path.basename(lCsvFileName[ii])

        if verifySingleSenderId:
            vSenderId = dfCurrData['Sender ID'].unique() #<! TODO: Merge it with the previous calculation
            numUniqueSenderId = len(vSenderId)
            sSenderId = ''
            for jj in range(numUniqueSenderId):
                sSenderId = f'Sender ID: {vSenderId[jj]}, '
            if numUniqueSenderId != 1:
                raise ValueError(f'The file {os.path.basename(lCsvFileName[ii])} has more than a single unique `Sender ID`: {sSenderId}')

        if verifyColumns:
            lColumns = dfCurrData.columns
            for colName in lColumns:
                if not(colName in lColName):#if not(colName in lCsvColName):
                    raise ValueError(f'The file {os.path.basename(lCsvFileName[ii])} has a column name: {colName} which is not defined')
            for jj, colName in enumerate(lColName):
                if lColFlag[jj] and (not(colName in lColumns)):#if lCsvColNameFlag[jj] and (not(colName in lColumns)):
                    raise ValueError(f'The file {os.path.basename(lCsvFileName[ii])} does not have the required column name: {colName}')

        dfData = pd.concat([dfData, dfCurrData], ignore_index = True)
    
    return dfData, dAssetFile


def JitterXData( xVal, numSamples, jitterLvl = 0.01 ):
    """
    Jitter of Scatter Data
    """
    return xVal + jitterLvl * np.random.randn(numSamples)


def ExtractPathParentFolder(parentFolder, fullFolferPath = os.getcwd()):
    # Don't use '\' at the end of `parentFolder`
    # TODO: Change this into regular expression search and last index (Something like parentFolder.replace('\\', '/').split('/'))
    pathHead = fullFolferPath
    pathTail = ' '
    while(pathTail != parentFolder):
        pathHead, pathTail = os.path.split(pathHead)
    return os.path.join(pathHead, pathTail)

def ExtractCsvFiles( parentDir, folderNamePattern = '' ) -> List:
    lCsvFiles = []
    for currDirPath, lDir, lFile in os.walk(parentDir, topdown = False): 
        for fileNameFull in lFile:
            fileName, fileExt = os.path.splitext(fileNameFull)
            folderPath, folderName = os.path.split(currDirPath)
            if (fileExt == '.csv') and (folderNamePattern in folderName):
                lCsvFiles.append(os.path.join(currDirPath, fileNameFull))
    
    return lCsvFiles

def CalcAttackType(dfData: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    # SASASW - Single Asset, Single   Attacks, Single   Wallets
    # SAMASW - Single Asset, Multiple Attacks, Single   Wallets (SAMA)
    # SAMAMW - Single Asset, Multiple Attacks, Multiple Wallets (SAMA)

    numGrps = dfData.GrpBySender.numGrps
    
    dsAttackType = pd.Series(index = dfData.index)
    dfAttackType = pd.DataFrame(data = np.empty((numGrps, 2)), columns = ['Asset ID', 'Attack Type'])
    for ii, (grpName, dfGrp) in enumerate(dfData.GrpBySender.dfGrpBySender):
        vIdx = dfGrp.index
        numUniqueRcv = dfGrp[dfGrp['Label'] == 1]['Receiver ID'].nunique()
        numLabeledTsx = (dfGrp['Label'] == 1).sum()
        
        if numLabeledTsx > 1:
            if numUniqueRcv > 1:
                #<! Multiple labeled transactions, Multiple receiver ID -> SASAMW
                attackType = AttackType.SAMAMW
            else:
                #<! Multiple labeled transactions, Single receiver ID -> SASASW
                attackType = AttackType.SAMASW
        else: #<! Single labeled transaction -> SASASW
            attackType = AttackType.SASASW
        
        dsAttackType[vIdx]              = attackType.value
        dfAttackType['Asset ID'][ii]    = grpName
        dfAttackType['Attack Type'][ii] = attackType.name

    return dsAttackType, dfAttackType


def GenClassifierSummaryResults( vY, vYPred ) -> pd.Series:

    lLabels = ['Accuracy', 'F1', 'Precision', 'Recall' , 'ROC AUC Score']
    lScores = [accuracy_score, f1_score, precision_score, recall_score , roc_auc_score]

    dScores = {}

    for ii, hScore in enumerate(lScores):
        dScores[lLabels[ii]] = hScore(vY, vYPred)
    
    return pd.Series(data = dScores)


def GenClassifierSummaryResultsGrp( vY, vYPred, vGrpLabel: np.ndarray, dGrpName: Dict = None ) -> pd.DataFrame:

    vGrpLabels = np.unique(vGrpLabel) #<! Sorted
    numLabels = len(vGrpLabels)

    if dGrpName is None:
        dGrpName = {}
        for ii, grpLabel in enumerate(vGrpLabels):
            dGrpName[grpLabel] = str(ii)

    lSummarySeries = []

    for ii, grpLabel in enumerate(vGrpLabels):
        grpName = dGrpName[grpLabel] #<! Assumed to be sorted
        vIdx = vGrpLabel == grpLabel
        vYTmp = vY[vIdx]
        vYPredTmp = vYPred[vIdx]

        dsRslt = GenClassifierSummaryResults(vYTmp, vYPredTmp)
        dfRslt = pd.DataFrame(dsRslt, columns = ['Score'])
        dfRslt['Group Name'] = grpName

        lSummarySeries.append(dfRslt)
    
    return pd.concat(lSummarySeries)

# Visualization

def DisplayConfusionMatrix( vY, vYPred, lClasses, hAx = None ) -> None:
    #TODO: Add support for the `hAx` input`
    mConfMat = confusion_matrix(vY, vYPred, labels = lClasses)
    cmDisp = ConfusionMatrixDisplay(confusion_matrix = mConfMat, display_labels = lClasses)

    cmPlot = cmDisp.plot()
    hA = cmPlot.ax_
    hA.grid(False)
    plt.show()

def DisplayConfusionMatrixGrp( vY, vYPred, lClasses, vGrpLabel: np.ndarray, lGrpName: List, mHAx = None ) -> None:

    vGrpLabel       = vGrpLabel.unique()
    numGrpLabels    = len(vGrpLabel)

    if mHAx.size < numGrpLabels:
        raise ValueError(f'The number of unique groups must be equal or smaller to the number of axis to plot on')

    # TODO: By @Anton

def DisplayScatterFeature( dfData, xColName, yColName, legendTitle, hA = None ) -> None:
    
    if hA is None:
        hF, hA = plt.subplots(figsize = (20, 10))
    
    sns.scatterplot(data = dfData, x = xColName, y = yColName, hue = xColName, ax = hA)
    hA.tick_params(top = False, bottom = False, labelbottom = False)
    hA.legend(title = 'Suspicious')
    plt.show()


def DisplayKdeFeature( dfData, colName, labelName, legendTitle, hA = None ) -> None:
    
    if hA is None:
        hF, hA = plt.subplots(figsize = (20, 10))
    
    sns.kdeplot(dfData[dfData[labelName] == 0][colName], ax = hA, color = 'b', label = '0')
    sns.kdeplot(dfData[dfData[labelName] == 1][colName], ax = hA, color = 'r', label = '1')

    hA.legend(title = legendTitle)
    plt.show()


def DisplayPairFeature( dfData, xColName, yColName, labelName, legendTitle, hA = None ) -> None:
    
    if hA is None:
        hF, hA = plt.subplots(figsize = (20, 10))
    
    sns.scatterplot(data = dfData, x = xColName, y = yColName, hue = labelName, style = labelName, ax = hA)

    hA.legend(title = legendTitle)
    plt.show()


def PreProcessData( dfData: pd.DataFrame, updateInplace = False, amountUsdOutlierThr = 1e9 ) -> pd.DataFrame:
    dfData['Block Time'] = pd.to_datetime(dfData['Block Time'], infer_datetime_format = 'True') #<! Stable time format

    if updateInplace:
        dfData.sort_values('Block Time', inplace = updateInplace)
    else:
        dfData = dfData.sort_values('Block Time', inplace = updateInplace) #<! Create a copy only at first operation (Is set to copy)

    dsInValidTrnsUsd = ((dfData['Amount [USD]'] == 0) | (dfData['Amount [USD]'].isna()) | (dfData['Amount [USD]'] == ''))
    dfData.drop(dfData.index[dsInValidTrnsUsd], inplace = True) #<! Royi: Should we do a reset index?
    
    dsOutlierTrnsUsd = ((dfData['Amount [USD]'] >= amountUsdOutlierThr) | (dfData['Amount [USD]'] <= 0))
    dfData.drop(dfData.index[dsOutlierTrnsUsd], inplace = True) #<! Royi: Should we do a reset index?

    return dfData


def TrainClassifier(dfGbs, feature_list : List, lCatFeatures: List):

    lNumericalFeatures = [featureName for featureName in feature_list if featureName not in lCatFeatures]
    lTotalFeatures = lNumericalFeatures + lCatFeatures
    
    
    dfGbs.dfData.replace([np.inf, -np.inf], np.nan, inplace = True)
    for i in feature_list:
        dfGbs.dfData[i].fillna(0, inplace = True)
    
    dfX = dfGbs.dfData[feature_list].copy()

    for catColName in lCatFeatures:
        if catColName in dfX.columns:
            dfX[catColName] = dfX[catColName].astype('category', copy = False)

    hStdScaler = StandardScaler()
    dfX[lNumericalFeatures] = hStdScaler.fit_transform(dfX[lNumericalFeatures])
    mX = dfX[lTotalFeatures]
    mX.rename(columns = {'Amount [USD]':'Amount USD'}, inplace = True)
    vY = dfGbs.dfData['Label']
    
    hKFoldSplt = StratifiedGroupKFold(n_splits = numKFolds, shuffle = True, random_state = randomState)
    for vTrainIdx, vTestIdx in hKFoldSplt.split(mX, vY, groups = dfGbs.dfData['Sender ID']):
        mXTrain, mXTest, vYTrain, vYTest = mX.iloc[vTrainIdx], mX.iloc[vTestIdx], vY.iloc[vTrainIdx], vY.iloc[vTestIdx]
        
        xgbModel =XGBClassifier(n_estimators=250, tree_method="hist", max_depth = 20,  random_state=seedNum, enable_categorical=True)
        xgbModel.fit(mXTrain, vYTrain)
        vYPred = xgbModel.predict(mXTest)
        DisplayConfusionMatrix(vYTest, vYPred, lClasses = xgbModel.classes_)
        print(GenClassifierSummaryResults(vYTest, vYPred))

    return xgbModel



def MergeTransIDTokens(df, lst):
    transIdStr = 'Transaction ID' ; currStr = 'Currency' ; amntStr = 'Amount [USD]'
    replace_dct_cnt = {} ; replace_dct_val = {} 
    for col in lst:
        replace_dct_cnt[col] = col+'_cnt'
        replace_dct_val[col] = col+'_val'
         
    ########groupbys for currencies counts and summation
    dfg_cnt = pd.DataFrame({'count' : df.groupby([transIdStr,currStr])[currStr].count()}).reset_index()
    dfg_val = pd.DataFrame({'sum' : df.groupby([transIdStr,currStr])[amntStr].sum()}).reset_index()
    ########groubys for overall counts and summation:
    dfg_cnt_all = pd.DataFrame({'count_all' : df.groupby([transIdStr])[currStr].count()}).reset_index()
    dfg_val_all = pd.DataFrame({'Total_Amount' : df.groupby([transIdStr])[amntStr].sum()}).reset_index()
    
    ### Label: if at least one transaction equals to 1, then entire group is 1
    dfg_label_all = pd.DataFrame({'Label' : df.groupby([transIdStr])['Label'].any().astype(int)}).reset_index()
    
    o_cnt = dfg_cnt[dfg_cnt[currStr].isin(lst)].pivot_table('count', [transIdStr], currStr).reset_index()
    o_val = dfg_val[dfg_val[currStr].isin(lst)].pivot_table('sum', [transIdStr], currStr).reset_index()
    
    ### proper columns names:
    o_cnt.rename(columns = replace_dct_cnt, inplace = True)
    o_val.rename(columns = replace_dct_val, inplace = True)
    
    ### merge individual currencies' counts and value summation with overall counts and summations:
    df_final = ft.reduce(lambda left, right: pd.merge(left, right, on=transIdStr), [o_cnt, o_val, dfg_cnt_all, dfg_val_all , dfg_label_all])
    df_final.fillna(0, inplace = True)
    return df_final     



def GenDataPredict(dfData , lSlctdFeatures , lNumericalFeatures , lCatFeatures):
    
    dfX = dfData[lSlctdFeatures].copy()
    dfX.replace([np.inf, -np.inf], np.nan, inplace = True)
    dfX.fillna(0, inplace = True)
    
    scaler_dct = {}
    for f in lNumericalFeatures:
        hStdScaler = StandardScaler()
        hStdScaler = hStdScaler.fit(dfX[[f]]) 
        dfX[f] = hStdScaler.transform(dfX[[f]])
        scaler_dct[f] = hStdScaler

    #hStdScaler = StandardScaler()
    #dfX[lNumericalFeatures] = hStdScaler.fit_transform(dfX[lNumericalFeatures])


    for catColName in lCatFeatures:
        if catColName in dfX.columns:
            dfX[catColName] = dfX[catColName].astype('category', copy = False)
    
    dfX.rename(columns = {'Amount [USD]': 'Amount USD'}, inplace = True)
    dfX['Label'] = dfData['Label']
    dfX['Sender ID']  =  dfData['Sender ID']
    
    return dfX , scaler_dct

def TrainModelByTransact(dfData ,lSlctdFeatures_  , lCatFeatures ,lSelectedFeatures_ , numKFolds, randomState ):
    
    models  = []

    dfX = dfData[lSlctdFeatures_].copy()
    
    for catColName in lCatFeatures:
        dfX[catColName], _ =  pd.factorize(dfX[catColName])
		
    lNumericalFeatures_ = [featureName for featureName in lSlctdFeatures_ if featureName not in lCatFeatures]

    mX = dfX[lNumericalFeatures_].to_numpy()
    mC = dfX[lCatFeatures].to_numpy()
    vY = dfData['Label'].to_numpy()
    mX = np.concatenate((mX, mC), axis = 1)



    # Training by Transactions (K-Fold)
    hKFoldSplt = StratifiedKFold(n_splits = numKFolds, shuffle = True, random_state = randomState)
    for vTrinIdx, vTestIdx in hKFoldSplt.split(mX, vY):
        mXTrain, mXTest, vYTrain, vYTest = mX[vTrinIdx, :], mX[vTestIdx, :], vY[vTrinIdx], vY[vTestIdx]
        xgbModel = XGBClassifier(use_label_encoder = False)
        xgbModel.fit(mXTrain, vYTrain)
        vYPred = xgbModel.predict(mXTest)
        DisplayConfusionMatrix(vYTest, vYPred, lClasses = xgbModel.classes_)
        print(GenClassifierSummaryResults(vYTest, vYPred))
        models.append([vTrinIdx, vTestIdx, xgbModel])

    return models

def TrainModelByFiles(dfData , lSelectedFeatures_ , numKFolds, randomState , seedNum):
        
    models  = []
    mX = dfData[lSelectedFeatures_]
    vY = dfData['Label']
    
    hKFoldSplt = StratifiedGroupKFold(n_splits = numKFolds, shuffle = True, random_state = randomState)
    for vTrainIdx, vTestIdx in hKFoldSplt.split(mX, vY, groups = dfData['Sender ID']):
        mXTrain, mXTest, vYTrain, vYTest = mX.iloc[vTrainIdx], mX.iloc[vTestIdx], vY.iloc[vTrainIdx], vY.iloc[vTestIdx]
        xgbModel =XGBClassifier(n_estimators=250, tree_method="hist", max_depth = 20,  random_state=seedNum, enable_categorical=True)
        xgbModel.fit(mXTrain, vYTrain)
        vYPred = xgbModel.predict(mXTest)
        DisplayConfusionMatrix(vYTest, vYPred, lClasses = xgbModel.classes_)
        print(GenClassifierSummaryResults(vYTest, vYPred))
        models.append([vTrainIdx, vTestIdx, xgbModel])

    return models  


def ValidateData(dfData, lSlctedFeaturesRaw) :

    lDfColName = dfData.columns.to_list()

    for colName in lSlctedFeaturesRaw:
        if colName not in lDfColName:
            raise ValueError(f'The column {colName} is missing in the data frame')


  
 
def hashfile(file):
  
    BUF_SIZE = 65536
    sha256 = hashlib.sha256()
  
    with open(file, 'rb') as f:
         
        while True:
             
            data = f.read(BUF_SIZE)
  
            if not data:
                break
      
            sha256.update(data)
  
      
    return sha256.hexdigest()