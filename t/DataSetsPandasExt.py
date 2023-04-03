# General Tools
import numpy as np
import scipy as sp
import pandas as pd

# Machine Learning
from sklearn.preprocessing import LabelEncoder

# Miscellaneous
from enum import Enum, auto, unique
import functools
# import random
import os
# import datetime
# from platform import python_version
import time

# Accelerators
from numba import njit

# Typing
from typing import List
import numba

@unique
class AmountType(Enum):
    # Type of data in the CSV
    AMOUNT_USD        = auto()
    AMOUNT_TOKEN      = auto()
    TIME_DIFF_ASSET   = auto()
    TIME_DIFF_USR     = auto()

@unique
class CalcType(Enum):
    TYPE_SUM                   = auto()
    TYPE_MEAN                  = auto()
    TYPE_STD                   = auto()
    TYPE_VAR                   = auto()
    TYPE_MEDIAN                = auto()
    TYPE_COUNT                 = auto()
    TYPE_MIN                   = auto()
    TYPE_MAX                   = auto()
    TYPE_PCTILE                = auto() #<! Quantile
    TYPE_TIME_DIFF_MEAN        = auto()
    TYPE_TIME_DIFF_STD         = auto()
    TYPE_TIME_DIFF_MEDIAN      = auto()
    TYPE_TIME_DIFF_MIN         = auto()
    TYPE_TIME_DIFF_MAX         = auto()
    TYPE_COUNT_COIN_TYPE       = auto()
    TYPE_COUNT_RECEIVER_TYPE   = auto()
    TYPE_COUNT_UNIQUE          = auto()


@unique
class PeriodTimeType(Enum):
    # Type of data in the CSV
    HOUR_DAY = auto()
    DAY_WEEK = auto()


D_AMOUNT_TYPE_COL_NAME = {AmountType.AMOUNT_USD: 'Amount [USD]', AmountType.AMOUNT_TOKEN: 'Amount'}

D_CALC_TYPE = {CalcType.TYPE_SUM                 : 'sum', 
               CalcType.TYPE_MEAN                : 'mean', 
               CalcType.TYPE_STD                 : 'std',
               CalcType.TYPE_VAR                 : 'var', 
               CalcType.TYPE_MEDIAN              : 'median', 
               CalcType.TYPE_COUNT               : 'count', 
               CalcType.TYPE_MIN                 : 'min',
               CalcType.TYPE_MAX                 : 'max', 
               CalcType.TYPE_PCTILE              : 'quantile',
               CalcType.TYPE_TIME_DIFF_MEAN      : 'mean',
               CalcType.TYPE_TIME_DIFF_STD       : 'std',
               CalcType.TYPE_TIME_DIFF_MEDIAN    : 'median',
               CalcType.TYPE_TIME_DIFF_MIN       : 'min',
               CalcType.TYPE_TIME_DIFF_MAX       : 'max',
               CalcType.TYPE_COUNT_COIN_TYPE     : 'nunique',
               CalcType.TYPE_COUNT_RECEIVER_TYPE : 'nunique',
               CalcType.TYPE_COUNT_UNIQUE        : 'nunique'
               }

def timer(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        elapsed_time = toc - tic
        print(f'Function {func.__name__!r} executed in {elapsed_time:0.4f} seconds')#print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return value
    return wrapper_timer

def aggregator(a, f):
    val = a.agg(f)
    return val

@pd.api.extensions.register_dataframe_accessor("GrpBySender")
class GrpBySender:
    #TODO: Make the `vGrpLabel` a property as well
    def __init__(self, dfData: pd.DataFrame):
        '''
        assume the data is always sorted by time var('Transaction Time')!
        remove value_sort() for time related !!!!
        '''
        grpColLabel         = 'Sender ID'
        usrColLabel         = 'Receiver ID'
        timeColLabel        = 'Block Time'
        currencyColLabel    = 'Currency'
        amountUSDColLabel   = 'Amount [USD]' 
        timeDiffAssetColLabel = 'TSX Time Diff (Asset)' 
        timeDiffUserColLabel = 'TSX Time Diff (User)'
        receiverTypeColLabel =  'Receiver Type' 
        gasPriceColLabel    =  'Gas Price'
        gasLimitColLabel    =  'Gas Limit'
        gasUsedColLabel     = 'Gas Used'
        self.grpColLabel    = grpColLabel
        self.usrColLabel    = usrColLabel
        self.currencyColLabel = currencyColLabel
        self.amountUSDColLabel   = amountUSDColLabel 
        self.timeDiffAssetColLabel = timeDiffAssetColLabel 
        self.timeDiffUserColLabel = timeDiffUserColLabel
        self.receiverTypeColLabel = receiverTypeColLabel
        self.gasPriceColLabel    =  gasPriceColLabel  
        self.gasLimitColLabel    =  gasLimitColLabel 
        self.gasUsedColLabel     =  gasUsedColLabel
        self._validate(dfData, grpColLabel, timeColLabel)
        self.timeColLabel   = timeColLabel
        self.dfData         = dfData
        self.dfGrpBySender  = dfData.groupby(grpColLabel) #<! GroupBy sorts data, so it should match LabelEncoder()
        
        # TODO: Add an index where grpLabel -> List of Indices
        self.labelEnc   = LabelEncoder() #TODO: Replace it with pd.factorize() to preserve order
        self.labelEnc.fit(dfData[grpColLabel])
        self.vGrpLabel  = self.labelEnc.transform(dfData[grpColLabel]) #<! By sorted order
        self.numGrps    = len(self.labelEnc.classes_) #<! Duplicate the setters
        self.lLabelIdx  = self._GenGrpLabelIdx(dfData, self.vGrpLabel, self.numGrps)

        self.dfSubGrpByRec          = dfData.groupby([grpColLabel, usrColLabel])
        self.lSubGrpUsrLabelIdx     = self._GenSubGrp(self.dfGrpBySender, usrColLabel)
        
        self.dfData[timeDiffAssetColLabel] = self.dfGrpBySender[timeColLabel].transform('diff')
        self.dfData[timeDiffUserColLabel]  = self.dfSubGrpByRec[timeColLabel].transform('diff')
        # Convert into seconds
        self.dfData[timeDiffAssetColLabel] = self.dfData[timeDiffAssetColLabel].dt.total_seconds()
        self.dfData[timeDiffUserColLabel]  = self.dfData[timeDiffUserColLabel].dt.total_seconds()

    
    @staticmethod
    def _validate(dfData: pd.DataFrame, grpColLabel, timeColLabel):
        if grpColLabel not in dfData.columns:
            raise AttributeError(f'The DF Must have {grpColLabel} as a column')
        if pd.to_datetime(dfData[timeColLabel], errors = 'coerce').isnull().any():
            raise AttributeError(f'The DF Must have the {timeColLabel} column with a valid date format')
        if dfData[timeColLabel].is_monotonic_increasing is not True:
            raise AttributeError(f'The DF Must have the {timeColLabel} column sorted in an increasing manner')
        
    
    @staticmethod
    def _GenGrpLabelIdx(dfData, vGrpLabel, numGrps):
        # Create a list / vector lLabelIdx such that lLabelIdx[ii] returns a list of the indices of the group in self.dfData
        lLabelIdx = [[] for ii in range(numGrps)]

        numTx = dfData.shape[0] #<! Number of transactions

        for ii in range(numTx):
            lLabelIdx[vGrpLabel[ii]].append(dfData.index[ii])
        
        return lLabelIdx
    

    @staticmethod
    def _GenSubGrp(dfGrp, grpByCol):
        #TODO: Check order fits the indices in `lLabelIdx`, If not, use the indices (It should match as `groupby()` sorts data)
        
        lT = [None] * len(dfGrp)

        for ii, (grpName, dfGroup) in enumerate(dfGrp):
            numUniqueUsr = len(dfGroup[grpByCol].unique())
            lT[ii] = [[] for jj in range(numUniqueUsr)]
            dfSubGrp = dfGroup.groupby(grpByCol)
            for jj, (subGrpName, dfSubGroup) in enumerate(dfSubGrp):
                lT[ii][jj] = dfSubGroup.index.to_list()
        
        return lT

    
    @property
    def numGrps(self):
        return self._numGrps
    
    @numGrps.setter
    def numGrps(self, setVal):
        self._numGrps = len(self.labelEnc.classes_)
    
    def AggBySender(self, colName, grpLabel = None, calcType = CalcType.TYPE_SUM):
        methodName = D_CALC_TYPE[calcType]

        if grpLabel is not None:
            return getattr(self.dfData[D_AMOUNT_TYPE_COL_NAME[colName]][self.lLabelIdx[grpLabel]], methodName)()
        else:
            return self.dfGrpBySender[colName].transform(methodName)#return self.dfGrpBySender[D_AMOUNT_TYPE_COL_NAME[colName]].transform(methodName)
    
    def AggByReceiver(self, colName, grpLabel = None, subGrpLabel = None, calcType = CalcType.TYPE_SUM):
        methodName = D_CALC_TYPE[calcType]

        if (grpLabel is not None) and (subGrpLabel is not None):
            return getattr(self.dfData[colName][self.lSubGrpUsrLabelIdx[grpLabel][subGrpLabel]], methodName)()
        else:
            return self.dfSubGrpByRec[colName].transform(methodName)
    
    def GetTimeVals(self, periodTimeType: PeriodTimeType) -> pd.Series:
        if periodTimeType == PeriodTimeType.HOUR_DAY:
            return self.dfData[self.timeColLabel].dt.hour
        elif periodTimeType == PeriodTimeType.DAY_WEEK:
            return self.dfData[self.timeColLabel].dt.dayofweek

    def AvgByUserCoinType(self):
        #56
        #Create features based on the currency of the transactions:
        # 1. The number of different types of currencies per user. <-- done previously = dfData[FeatureName.COIN_TYPE_COUNT_USR.name]
        # 2. The average of the number of types of all user for an asset. <-- groupby asset , mean(number of different types of currencies per user)
        # 3. The ratio between a specific user to the average of the asset. --> 1/2
            
        self.dfData['COIN_TYPE_COUNT_USR'] = self.AggByReceiver(colName = self.currencyColLabel, grpLabel = None, calcType = CalcType.TYPE_COUNT_COIN_TYPE) #df[COIN_TYPE_COUNT_USR.name] = df['coin_count']   = df.groupby(['asset','user'])['coin'].transform('nunique')
        self.dfData['tsx_count']     = self.AggByReceiver(colName = self.usrColLabel, grpLabel = None, calcType = CalcType.TYPE_COUNT)  #df['tsx_count']     = df.groupby(['asset','user'])['user'].transform('count')
        self.dfData['num_usr_asset'] = self.AggBySender(colName = self.usrColLabel, grpLabel = None, calcType = CalcType.TYPE_COUNT_UNIQUE)#df['num_usr_asset'] = df.groupby(['asset'])['user'].transform('nunique')
        self.dfData['coin_count/tsx_count'] = self.dfData['COIN_TYPE_COUNT_USR'] / self.dfData['tsx_count']#df['coin_count/tsx_count'] = df['coin_count'] / df['tsx_count']
        self.dfData['usr_sum'] = self.AggBySender(colName = 'coin_count/tsx_count', grpLabel = None, calcType = CalcType.TYPE_SUM) #df['usr_sum'] = df.groupby(['asset'])['coin_count/tsx_count'].transform('sum')
        valDs = self.dfData['usr_sum'] / self.dfData['num_usr_asset']
        self.dfData.drop(['tsx_count', 'num_usr_asset', 'coin_count/tsx_count' , 'usr_sum'], axis=1 , inplace=True) #<<-- remove temporary columns
        return valDs   #self.dfData[FeatureName.COIN_TYPE_COUNT_USR_MEAN_ASSET.name]
        #return dfData[FeatureName.COIN_TYPE_COUNT_USR.name] / dfData[FeatureName.COIN_TYPE_COUNT_USR_MEAN_ASSET.name]
                                                            


"""
Some kind of unit testing (We need to do something like that)

# Sort data by transaction date
dfData.sort_values('Transaction Time', inplace = True)

# Verify group by keeps the sorted values

dfGrpBySender = dfData.groupby('Sender ID')
for grpName, dfGroup in dfGrpBySender:
    if dfGroup['Transaction Time'].is_monotonic_increasing is not True:
        raise ValueError(f'The sub group {grpName} is not sorted')

for grpName, dfGroup in dfGrpBySender:
    dfSubGrpBy = dfGroup.groupby('Receiver ID')
    for subGrpName, dfSubGroup in dfSubGrpBy:
        if dfSubGroup['Transaction Time'].is_monotonic_increasing is not True:
            raise ValueError(f'The sub sub group {subGrpName} of the sub group {grpName} is not sorted')

dfGrpBySenderReceiver = dfData.groupby(['Sender ID', 'Receiver ID'])

for grpName, dfGroup in dfGrpBySenderReceiver:
    if dfGroup['Transaction Time'].is_monotonic_increasing is not True:
        raise ValueError(f'The sub group {grpName} is not sorted')
"""
















