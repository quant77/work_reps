# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Machine Learning
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Miscellaneous
from enum import Enum, auto, unique
# import random
import os
# import datetime
# from platform import python_version

# Typing
from typing import List

# Visualization
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from bokeh.plotting import figure, show

from DataSetsPandasExt import *

lCsvColName     = ['Transaction ID', 'Transaction Time', 'Sender ID', 'Receiver ID', 'Amount', 'Currency', 'Currency Hash', 'Currency Type', 'Amount [USD]', 'Gas Price', 'Gas Limit', 'Gas Used', 'Gas Predicted', 'Receiver Type', 'Label']
lCsvColNameFlag = [True,              True,              True,        True,           True,     True,       True,            True,            True,           False,        False,       False,      False,           True,            False] #<! Flags if a column is a must to have


def LoadCsvFilesDf( lCsvFileName: List, baseFoldePath = '.', verifySingleSenderId = True, verifyColumns = True, lColName = lCsvColName, lColFlag =  lCsvColNameFlag, addFileNameCol = False ) -> pd.DataFrame:

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
            if not(colName in lCsvColName):
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
                if not(colName in lCsvColName):
                    raise ValueError(f'The file {os.path.basename(lCsvFileName[ii])} has a column name: {colName} which is not defined')
            for jj, colName in enumerate(lColName):
                if lCsvColNameFlag[jj] and (not(colName in lColumns)):
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


def GenClassifierSummaryResults( vY, vYPred ) -> pd.Series:

    lLabels = ['Accuracy', 'F1', 'Precision', 'Recall']
    lScores = [accuracy_score, f1_score, precision_score, recall_score]

    dScores = {}

    for ii, hScore in enumerate(lScores):
        dScores[lLabels[ii]] = hScore(vY, vYPred)
    
    return pd.Series(data = dScores)

# Visualization

def DisplayScatterFeature( dfData, xColName, yColName, legendTitle, hA = None ):
    
    if hA is None:
        hF, hA = plt.subplots(figsize = (20, 10))
    
    sns.scatterplot(data = dfData, x = xColName, y = yColName, hue = xColName, ax = hA)
    hA.tick_params(top = False, bottom = False, labelbottom = False)
    hA.legend(title = 'Suspicious')
    plt.show()


def DisplayKdeFeature( dfData, colName, labelName, legendTitle, hA = None ):
    
    if hA is None:
        hF, hA = plt.subplots(figsize = (20, 10))
    
    sns.kdeplot(dfData[dfData[labelName] == 0][colName], ax = hA, color = 'b', label = '0')
    sns.kdeplot(dfData[dfData[labelName] == 1][colName], ax = hA, color = 'r', label = '1')

    hA.legend(title = legendTitle)
    plt.show()