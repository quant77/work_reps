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

# Typing
from typing import List
import numba

@unique
class AmountType(Enum):
    # Type of data in the CSV
    AMOUNT_USD        = auto()
    AMOUNT_TOKEN      = auto()

@unique
class CalcType(Enum):
    TYPE_SUM                = auto()
    TYPE_MEAN               = auto()
    TYPE_STD                = auto()
    TYPE_VAR                = auto()
    TYPE_MEDIAN             = auto()
    TYPE_COUNT              = auto()
    TYPE_MIN                = auto()
    TYPE_MAX                = auto()
    TYPE_PCTILE             = auto() #<! Quantile
    TYPE_TIME_DIFF_MEAN     = auto()
    TYPE_TIME_DIFF_STD      = auto()
    TYPE_TIME_DIFF_MEDIAN   = auto()
    TYPE_TIME_DIFF_MIN      = auto()
    TYPE_TIME_DIFF_MAX      = auto()
    TYPE_COUNT_COIN_TYPE    = auto()




D_AMOUNT_TYPE_COL_NAME = {AmountType.AMOUNT_USD: 'Amount [USD]', AmountType.AMOUNT_TOKEN: 'Amount'}
D_CALC_TYPE = {CalcType.TYPE_SUM: 'sum', 
               CalcType.TYPE_MEAN: 'mean', 
               CalcType.TYPE_STD: 'std',
               CalcType.TYPE_VAR: 'var', 
               CalcType.TYPE_MEDIAN: 'median', 
               CalcType.TYPE_COUNT: 'count', 
               CalcType.TYPE_MIN: 'min',
               CalcType.TYPE_MAX: 'max', 
               CalcType.TYPE_PCTILE: 'quantile',
               CalcType.TYPE_TIME_DIFF_MEAN     : 'mean',
               CalcType.TYPE_TIME_DIFF_STD      : 'std',
               CalcType.TYPE_TIME_DIFF_MEDIAN   : 'median',
               CalcType.TYPE_TIME_DIFF_MIN      : 'min',
               CalcType.TYPE_TIME_DIFF_MAX      : 'max'
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

@numba.jit
def aggregator(a, f):
    val = a.agg(f)
    return val

@pd.api.extensions.register_dataframe_accessor("GrpBySender")
class GrpBySender:
    #TODO: Make the `vGrpLabel` a property as well
    #TODO: Make a group by of receiver per asset
    #TODO: Validate: Data is ordered by TX Time, Date Time column is valid Panda format
    def __init__(self, dfData: pd.DataFrame):
        '''
        assume the data is always sorted by time var('Transaction Time')!
        remove value_sort() for time related !!!!
        '''
        grpColLabel         = 'Sender ID'
        usrColLabel         = 'Receiver ID'
        timeColLabel        = 'Transaction Time'
        currencyColLabel    = 'Currency'
        self.grpColLabel    = grpColLabel
        self.currencyColLabel = currencyColLabel
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

        self.lSubGrpUsrLabelIdx  = self._GenSubGrp(self.dfGrpBySender, usrColLabel)
        # self.dfData[self.timeColLabel] = pd.to_datetime(self.dfData[self.timeColLabel], infer_datetime_format = 'True') #<! @Anton: I think it won't change the user DF. So it might create a fork.

    
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
                lT[ii][jj].append(dfSubGroup.index.to_list())
        
        return lT

    
    @property
    def numGrps(self):
        return self._numGrps
    
    @numGrps.setter
    def numGrps(self, setVal):
        self._numGrps = len(self.labelEnc.classes_)
    
    def TotalSentValue(self, amountCol: AmountType = AmountType.AMOUNT_USD, tokenId = None, grpLabel = None): 
        #Total Value in ETH sent by a wallet
       
       return self._SentValue(amountCol = amountCol, tokenId = tokenId, grpLabel = grpLabel, calcType = CalcType.TYPE_SUM)
    
    def AvgSentValue(self, amountCol: AmountType = AmountType.AMOUNT_USD, tokenId = None, grpLabel = None): 
        #Average Value in ETH sent by a wallet
        
        return self._SentValue(amountCol = amountCol, tokenId = tokenId, grpLabel = grpLabel, calcType = CalcType.TYPE_AVG)
    
    def StdSentValue(self, amountCol: AmountType = AmountType.AMOUNT_USD, tokenId = None, grpLabel = None): 
        #Average Value in ETH sent by a wallet
        
        return self._SentValue(amountCol = amountCol, tokenId = tokenId, grpLabel = grpLabel, calcType = CalcType.TYPE_STD)
    
    def VarSentValue(self, amountCol: AmountType = AmountType.AMOUNT_USD, tokenId = None, grpLabel = None): 
        #Average Value in ETH sent by a wallet
        
        return self._SentValue(amountCol = amountCol, tokenId = tokenId, grpLabel = grpLabel, calcType = CalcType.TYPE_VAR)
    
    def MedianSentValue(self, amountCol: AmountType = AmountType.AMOUNT_USD, tokenId = None, grpLabel = None): 
        #Average Value in ETH sent by a wallet
        
        return self._SentValue(amountCol = amountCol, tokenId = tokenId, grpLabel = grpLabel, calcType = CalcType.TYPE_MEDIAN)
    
    def MaxSentValue(self, amountCol: AmountType = AmountType.AMOUNT_USD, tokenId = None, grpLabel = None): 
        #Average Value in ETH sent by a wallet
        
        return self._SentValue(amountCol = amountCol, tokenId = tokenId, grpLabel = grpLabel, calcType = CalcType.TYPE_MAX)
    
    def MinSentValue(self, amountCol: AmountType = AmountType.AMOUNT_USD, tokenId = None, grpLabel = None): 
        #Average Value in ETH sent by a wallet
        
        return self._SentValue(amountCol = amountCol, tokenId = tokenId, grpLabel = grpLabel, calcType = CalcType.TYPE_MIN)

    @timer
    def _SentValue(self, amountCol: AmountType = AmountType.AMOUNT_USD, tokenId = None, grpLabel = None, calcType: CalcType = CalcType.TYPE_SUM):
        # TODO: Add support for enquiring specific token
        # TODO: Verify it vs. Anton Code
        # TODO: Make it one line using return dfData.agg(D_CALC_TYPE[CalcType.TYPE_SUM]) (Maybe better using a Lambda Function)

        if grpLabel is not None:
            # Return a single value (Scalar)
            if calcType == CalcType.TYPE_SUM:
                return self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lLabelIdx[grpLabel]].sum()
            elif calcType == CalcType.TYPE_MEAN:
                return self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lLabelIdx[grpLabel]].mean()
            elif calcType == CalcType.TYPE_STD:
                return self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lLabelIdx[grpLabel]].std()
            elif calcType == CalcType.TYPE_VAR:
                return self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lLabelIdx[grpLabel]].var()
            elif calcType == CalcType.TYPE_MEDIAN:
                return self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lLabelIdx[grpLabel]].median()
            elif calcType == CalcType.TYPE_MAX:
                return self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lLabelIdx[grpLabel]].max()
            elif calcType == CalcType.TYPE_MIN:
                return self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lLabelIdx[grpLabel]].min()
            elif calcType == CalcType.TYPE_COUNT:
                return self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lLabelIdx[grpLabel]].count()
            elif calcType == CalcType.TYPE_TIME_DIFF_MEAN:
                return self.dfData[self.timeColLabel][self.lLabelIdx[grpLabel]].sort_values().diff().dt.total_seconds().mean()
            elif calcType == CalcType.TYPE_TIME_DIFF_STD:
                return self.dfData[self.timeColLabel][self.lLabelIdx[grpLabel]].sort_values().diff().dt.total_seconds().std() 
            elif calcType == CalcType.TYPE_TIME_DIFF_MEDIAN:
                return self.dfData[self.timeColLabel][self.lLabelIdx[grpLabel]].sort_values().diff().dt.total_seconds().median()
            elif calcType == CalcType.TYPE_TIME_DIFF_MIN:
                return self.dfData[self.timeColLabel][self.lLabelIdx[grpLabel]].sort_values().diff().dt.total_seconds().min()
            elif calcType == CalcType.TYPE_TIME_DIFF_MEAN:
                return self.dfData[self.timeColLabel][self.lLabelIdx[grpLabel]].sort_values().diff().dt.total_seconds().max()               
            
        else:
            # Work on all data, return a series
            ds_SentValue = pd.Series(index = self.dfData.index) #<! Maybe better just do dsTotalSentValue = self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]] ? Then data is local in cache
            if calcType == CalcType.TYPE_SUM:
                for ii in range(self.numGrps):
                    ds_SentValue[self.lLabelIdx[ii]] = self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lLabelIdx[ii]].sum()
            elif calcType == CalcType.TYPE_MEAN:
                for ii in range(self.numGrps):
                    ds_SentValue[self.lLabelIdx[ii]] = self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lLabelIdx[ii]].mean()
            elif calcType == CalcType.TYPE_STD:
                for ii in range(self.numGrps):
                    ds_SentValue[self.lLabelIdx[ii]] = self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lLabelIdx[ii]].std()
            elif calcType == CalcType.TYPE_VAR:
                for ii in range(self.numGrps):
                    ds_SentValue[self.lLabelIdx[ii]] = self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lLabelIdx[ii]].var()
            elif calcType == CalcType.TYPE_MEDIAN:
                for ii in range(self.numGrps):
                    ds_SentValue[self.lLabelIdx[ii]] = self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lLabelIdx[ii]].median()
            elif calcType == CalcType.TYPE_MAX:
                for ii in range(self.numGrps):
                    ds_SentValue[self.lLabelIdx[ii]] = self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lLabelIdx[ii]].max()
            elif calcType == CalcType.TYPE_MIN:
                for ii in range(self.numGrps):
                    ds_SentValue[self.lLabelIdx[ii]] = self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lLabelIdx[ii]].min()
            elif calcType == CalcType.TYPE_COUNT:
                for ii in range(self.numGrps):
                    ds_SentValue[self.lLabelIdx[ii]] = self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lLabelIdx[ii]].count()
            elif calcType == CalcType.TYPE_COUNT_COIN_TYPE:
                for ii in range(self.numGrps):
                    ds_SentValue[self.lLabelIdx[ii]] = self.dfData[self.currencyColLabel][self.lLabelIdx[ii]].count()        
            
            # TODO: Move to a different, Time focused, function (This is Amount focused function)
            elif calcType == CalcType.TYPE_TIME_DIFF_MEAN:
                for ii in range(self.numGrps):
                    ds_SentValue[self.lLabelIdx[ii]] = self.dfData[self.timeColLabel][self.lLabelIdx[ii]].sort_values().diff().dt.total_seconds().mean()

            elif calcType == CalcType.TYPE_TIME_DIFF_STD:
                for ii in range(self.numGrps):
                    ds_SentValue[self.lLabelIdx[ii]] = self.dfData[self.timeColLabel][self.lLabelIdx[ii]].sort_values().diff().dt.total_seconds().std()

            elif calcType == CalcType.TYPE_TIME_DIFF_MEDIAN:
                for ii in range(self.numGrps):
                    ds_SentValue[self.lLabelIdx[ii]] = self.dfData[self.timeColLabel][self.lLabelIdx[ii]].sort_values().diff().dt.total_seconds().median()                
            
            elif calcType == CalcType.TYPE_TIME_DIFF_MIN:
                for ii in range(self.numGrps):
                    ds_SentValue[self.lLabelIdx[ii]] = self.dfData[self.timeColLabel][self.lLabelIdx[ii]].sort_values().diff().dt.total_seconds().min()
            
            elif calcType == CalcType.TYPE_TIME_DIFF_MAX:
                for ii in range(self.numGrps):
                    ds_SentValue[self.lLabelIdx[ii]] = self.dfData[self.timeColLabel][self.lLabelIdx[ii]].sort_values().diff().dt.total_seconds().max()

            return ds_SentValue
            

    @timer
    def _AnalyseRecieverId(self, amountCol = AmountType.AMOUNT_USD, tokenId = None, grpLabel = None, subGrpLabel = None, calcType = CalcType.TYPE_SUM):
        # Template for the analysis of Receiver
        # If grpLabel is

        if (grpLabel is not None) and (subGrpLabel is not None):
            if calcType == CalcType.TYPE_SUM:
                return self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[grpLabel][subGrpLabel][0]].sum()
            if calcType == CalcType.TYPE_MEAN:
                return self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[grpLabel][subGrpLabel][0]].mean()
            if calcType == CalcType.TYPE_STD:
                return self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[grpLabel][subGrpLabel][0]].std()    
            if calcType == CalcType.TYPE_VAR:
                return self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[grpLabel][subGrpLabel][0]].var()
            if calcType == CalcType.TYPE_MEDIAN:
                return self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[grpLabel][subGrpLabel][0]].median()
            if calcType == CalcType.TYPE_MAX:
                return self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[grpLabel][subGrpLabel][0]].max() 
            if calcType == CalcType.TYPE_MIN:
                return self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[grpLabel][subGrpLabel][0]].min()   
            if calcType == CalcType.TYPE_COUNT:
                return self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[grpLabel][subGrpLabel][0]].count()
            if calcType == CalcType.TYPE_TIME_DIFF_MEAN:
                return self.dfData[self.timeColLabel][self.lSubGrpUsrLabelIdx[grpLabel][subGrpLabel][0]].sort_values().diff().dt.total_seconds().mean()  
            if calcType == CalcType.TYPE_TIME_DIFF_STD:
                return self.dfData[self.timeColLabel][self.lSubGrpUsrLabelIdx[grpLabel][subGrpLabel][0]].sort_values().diff().dt.total_seconds().std()
            if calcType == CalcType.TYPE_TIME_DIFF_MAX:
                return self.dfData[self.timeColLabel][self.lSubGrpUsrLabelIdx[grpLabel][subGrpLabel][0]].sort_values().diff().dt.total_seconds().max()
            if calcType == CalcType.TYPE_TIME_DIFF_MIN:
                return self.dfData[self.timeColLabel][self.lSubGrpUsrLabelIdx[grpLabel][subGrpLabel][0]].sort_values().diff().dt.total_seconds().min()
            if calcType == CalcType.TYPE_TIME_DIFF_MEDIAN:
                return self.dfData[self.timeColLabel][self.lSubGrpUsrLabelIdx[grpLabel][subGrpLabel][0]].sort_values().diff().dt.total_seconds().median()        

        elif (grpLabel is None) and (subGrpLabel is not None):
            ds_SentValue = pd.Series(index = range(len(self.lSubGrpUsrLabelIdx)))
            if calcType == CalcType.TYPE_SUM:
                for ii in range(len(self.lSubGrpUsrLabelIdx)):
                    #ds_SentValue[self.lSubGrpUsrLabelIdx[ii][subGrpLabel][0]] =self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[ii][subGrpLabel][0]].sum()
                    ds_SentValue[ii] =self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[ii][subGrpLabel][0]].sum()
            if calcType == CalcType.TYPE_MEAN:
                for ii in range(len(self.lSubGrpUsrLabelIdx)):
                    ds_SentValue[ii] =self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[ii][subGrpLabel][0]].mean()
            if calcType == CalcType.TYPE_STD:
                for ii in range(len(self.lSubGrpUsrLabelIdx)):
                    ds_SentValue[ii] = self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[ii][subGrpLabel][0]].std()
            if calcType == CalcType.TYPE_VAR:
                for ii in range(len(self.lSubGrpUsrLabelIdx)):
                    ds_SentValue[ii] = self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[ii][subGrpLabel][0]].var() 
            if calcType == CalcType.TYPE_MEDIAN:
                for ii in range(len(self.lSubGrpUsrLabelIdx)):
                    ds_SentValue[ii] = self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[ii][subGrpLabel][0]].median() 
            if calcType == CalcType.TYPE_MIN:
                for ii in range(len(self.lSubGrpUsrLabelIdx)):
                    ds_SentValue[ii] = self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[ii][subGrpLabel][0]].min() 
            if calcType == CalcType.TYPE_MAX:
                for ii in range(len(self.lSubGrpUsrLabelIdx)):
                    ds_SentValue[ii] = self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[ii][subGrpLabel][0]].max() 
            if calcType == CalcType.TYPE_COUNT:
                for ii in range(len(self.lSubGrpUsrLabelIdx)):
                    ds_SentValue[ii] = self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[ii][subGrpLabel][0]].count()
            if calcType == CalcType.TYPE_TIME_DIFF_MEAN:
                for ii in range(len(self.lSubGrpUsrLabelIdx)):
                    ds_SentValue[ii] = self.dfData[self.timeColLabel][self.lSubGrpUsrLabelIdx[ii][subGrpLabel][0]].sort_values().diff().dt.total_seconds().mean()
            if calcType == CalcType.TYPE_TIME_DIFF_STD:
                for ii in range(len(self.lSubGrpUsrLabelIdx)):
                    ds_SentValue[ii] = self.dfData[self.timeColLabel][self.lSubGrpUsrLabelIdx[ii][subGrpLabel][0]].sort_values().diff().dt.total_seconds().std()
            if calcType == CalcType.TYPE_TIME_DIFF_MEDIAN:
                for ii in range(len(self.lSubGrpUsrLabelIdx)):
                    ds_SentValue[ii] = self.dfData[self.timeColLabel][self.lSubGrpUsrLabelIdx[ii][subGrpLabel][0]].sort_values().diff().dt.total_seconds().median()
            if calcType == CalcType.TYPE_TIME_DIFF_MAX:
                for ii in range(len(self.lSubGrpUsrLabelIdx)):
                    ds_SentValue[ii] = self.dfData[self.timeColLabel][self.lSubGrpUsrLabelIdx[ii][subGrpLabel][0]].sort_values().diff().dt.total_seconds().max()
            if calcType == CalcType.TYPE_TIME_DIFF_MIN:
                for ii in range(len(self.lSubGrpUsrLabelIdx)):
                    ds_SentValue[ii] = self.dfData[self.timeColLabel][self.lSubGrpUsrLabelIdx[ii][subGrpLabel][0]].sort_values().diff().dt.total_seconds().min()

            return ds_SentValue
        
        elif (grpLabel is not None) and (subGrpLabel is None):
            ds_SentValue = pd.Series(index = range(len(self.lSubGrpUsrLabelIdx[grpLabel])))
            if calcType == CalcType.TYPE_SUM:
                for ii in range(len(self.lSubGrpUsrLabelIdx[grpLabel])):
                    ds_SentValue[ii] = self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[grpLabel][ii][0]].sum()    
            if calcType == CalcType.TYPE_MEAN:
                for ii in range(len(self.lSubGrpUsrLabelIdx[grpLabel])):
                    ds_SentValue[ii] = self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[grpLabel][ii][0]].mean()
            if calcType == CalcType.TYPE_STD:
                for ii in range(len(self.lSubGrpUsrLabelIdx[grpLabel])):
                    ds_SentValue[ii] = self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[grpLabel][ii][0]].std()        
            if calcType == CalcType.TYPE_MEDIAN:
                for ii in range(len(self.lSubGrpUsrLabelIdx[grpLabel])):
                    ds_SentValue[ii] = self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[grpLabel][ii][0]].median()
            if calcType == CalcType.TYPE_VAR:
                for ii in range(len(self.lSubGrpUsrLabelIdx[grpLabel])):
                    ds_SentValue[ii] = self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[grpLabel][ii][0]].var()
            if calcType == CalcType.TYPE_MAX:
                for ii in range(len(self.lSubGrpUsrLabelIdx[grpLabel])):
                    ds_SentValue[ii] = self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[grpLabel][ii][0]].max()
            if calcType == CalcType.TYPE_MIN:
                for ii in range(len(self.lSubGrpUsrLabelIdx[grpLabel])):
                    ds_SentValue[ii] = self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[grpLabel][ii][0]].min()                      
            if calcType == CalcType.TYPE_COUNT:
                for ii in range(len(self.lSubGrpUsrLabelIdx[grpLabel])):
                    ds_SentValue[ii] = self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[grpLabel][ii][0]].count()
            if calcType == CalcType.TYPE_TIME_DIFF_MEAN:
                for ii in range(len(self.lSubGrpUsrLabelIdx[grpLabel])):
                    ds_SentValue[ii] = self.dfData[self.timeColLabel][self.lSubGrpUsrLabelIdx[grpLabel][ii][0]].sort_values().diff().dt.total_seconds().mean()
            if calcType == CalcType.TYPE_TIME_DIFF_STD:
                for ii in range(len(self.lSubGrpUsrLabelIdx[grpLabel])):
                    ds_SentValue[ii] = self.dfData[self.timeColLabel][self.lSubGrpUsrLabelIdx[grpLabel][ii][0]].sort_values().diff().dt.total_seconds().std()
            if calcType == CalcType.TYPE_TIME_DIFF_MEDIAN:
                for ii in range(len(self.lSubGrpUsrLabelIdx[grpLabel])):
                    ds_SentValue[ii] = self.dfData[self.timeColLabel][self.lSubGrpUsrLabelIdx[grpLabel][ii][0]].sort_values().diff().dt.total_seconds().median()
            if calcType == CalcType.TYPE_TIME_DIFF_MAX:
                for ii in range(len(self.lSubGrpUsrLabelIdx[grpLabel])):
                    ds_SentValue[ii] = self.dfData[self.timeColLabel][self.lSubGrpUsrLabelIdx[grpLabel][ii][0]].sort_values().diff().dt.total_seconds().max() 
            if calcType == CalcType.TYPE_TIME_DIFF_MIN:
                for ii in range(len(self.lSubGrpUsrLabelIdx[grpLabel])):
                    ds_SentValue[ii] = self.dfData[self.timeColLabel][self.lSubGrpUsrLabelIdx[grpLabel][ii][0]].sort_values().diff().dt.total_seconds().min()                  

            return ds_SentValue
    
        elif (subGrpLabel is None) and (subGrpLabel is None): #<! Equivalent of `else` but clearer
            ds_SentValue = pd.Series(index = self.dfData.index)
            if calcType == CalcType.TYPE_SUM:
                for ii in range(len(self.lSubGrpUsrLabelIdx)):
                    for i in range(len(self.lSubGrpUsrLabelIdx[ii])):
                        ds_SentValue[self.lSubGrpUsrLabelIdx[ii][i][0]] =self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[ii][i][0]].sum()

            if calcType == CalcType.TYPE_STD:
                for ii in range(len(self.lSubGrpUsrLabelIdx)):
                    for i in range(len(self.lSubGrpUsrLabelIdx[ii])):
                        ds_SentValue[self.lSubGrpUsrLabelIdx[ii][i][0]] =self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[ii][i][0]].std()            
            
            if calcType == CalcType.TYPE_MEAN:
                for ii in range(len(self.lSubGrpUsrLabelIdx)):
                    for i in range(len(self.lSubGrpUsrLabelIdx[ii])):
                        ds_SentValue[self.lSubGrpUsrLabelIdx[ii][i][0]] =self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[ii][i][0]].mean()

            if calcType == CalcType.TYPE_VAR:
                for ii in range(len(self.lSubGrpUsrLabelIdx)):
                    for i in range(len(self.lSubGrpUsrLabelIdx[ii])):
                        ds_SentValue[self.lSubGrpUsrLabelIdx[ii][i][0]] =self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[ii][i][0]].var()

            if calcType == CalcType.TYPE_MEDIAN:
                for ii in range(len(self.lSubGrpUsrLabelIdx)):
                    for i in range(len(self.lSubGrpUsrLabelIdx[ii])):
                        ds_SentValue[self.lSubGrpUsrLabelIdx[ii][i][0]] =self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[ii][i][0]].median()

            if calcType == CalcType.TYPE_MAX:
                for ii in range(len(self.lSubGrpUsrLabelIdx)):
                    for i in range(len(self.lSubGrpUsrLabelIdx[ii])):
                        ds_SentValue[self.lSubGrpUsrLabelIdx[ii][i][0]] =self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[ii][i][0]].max()
            
            if calcType == CalcType.TYPE_MIN:
                for ii in range(len(self.lSubGrpUsrLabelIdx)):
                    for i in range(len(self.lSubGrpUsrLabelIdx[ii])):
                        ds_SentValue[self.lSubGrpUsrLabelIdx[ii][i][0]] =self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[ii][i][0]].min()

            if calcType == CalcType.TYPE_COUNT:
                for ii in range(len(self.lSubGrpUsrLabelIdx)):
                    for i in range(len(self.lSubGrpUsrLabelIdx[ii])):
                        ds_SentValue[self.lSubGrpUsrLabelIdx[ii][i][0]] =self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[ii][i][0]].count()
            
            elif calcType == CalcType.TYPE_TIME_DIFF_MEAN:
                for ii in range(len(self.lSubGrpUsrLabelIdx)):
                    for i in range(len(self.lSubGrpUsrLabelIdx[ii])):
                        ds_SentValue[self.lSubGrpUsrLabelIdx[ii][i][0]] =self.dfData[self.timeColLabel][self.lSubGrpUsrLabelIdx[ii][i][0]].sort_values().diff().dt.total_seconds().mean() 

            elif calcType == CalcType.TYPE_TIME_DIFF_STD:
                for ii in range(len(self.lSubGrpUsrLabelIdx)):
                    for i in range(len(self.lSubGrpUsrLabelIdx[ii])):
                        ds_SentValue[self.lSubGrpUsrLabelIdx[ii][i][0]] =self.dfData[self.timeColLabel][self.lSubGrpUsrLabelIdx[ii][i][0]].sort_values().diff().dt.total_seconds().std() 

            elif calcType == CalcType.TYPE_TIME_DIFF_MEDIAN:
                for ii in range(len(self.lSubGrpUsrLabelIdx)):
                    for i in range(len(self.lSubGrpUsrLabelIdx[ii])):
                        ds_SentValue[self.lSubGrpUsrLabelIdx[ii][i][0]] =self.dfData[self.timeColLabel][self.lSubGrpUsrLabelIdx[ii][i][0]].sort_values().diff().dt.total_seconds().median() 
            
            elif calcType == CalcType.TYPE_TIME_DIFF_MAX:
                for ii in range(len(self.lSubGrpUsrLabelIdx)):
                    for i in range(len(self.lSubGrpUsrLabelIdx[ii])):
                        ds_SentValue[self.lSubGrpUsrLabelIdx[ii][i][0]] =self.dfData[self.timeColLabel][self.lSubGrpUsrLabelIdx[ii][i][0]].sort_values().diff().dt.total_seconds().max() 

            elif calcType == CalcType.TYPE_TIME_DIFF_MIN:
                for ii in range(len(self.lSubGrpUsrLabelIdx)):
                    for i in range(len(self.lSubGrpUsrLabelIdx[ii])):
                        ds_SentValue[self.lSubGrpUsrLabelIdx[ii][i][0]] =self.dfData[self.timeColLabel][self.lSubGrpUsrLabelIdx[ii][i][0]].sort_values().diff().dt.total_seconds().min() 
            

            elif calcType == CalcType.TYPE_COUNT_COIN_TYPE:
                for ii in range(len(self.lSubGrpUsrLabelIdx)):
                    for i in range(len(self.lSubGrpUsrLabelIdx[ii])):
                        ds_SentValue[self.lSubGrpUsrLabelIdx[ii][i][0]] =self.dfData[self.currencyColLabel][self.lSubGrpUsrLabelIdx[ii][i][0]].count()
                        

            return ds_SentValue
    
    @timer
    def _SentValue_new(self, amountCol: AmountType = AmountType.AMOUNT_USD, tokenId = None, grpLabel = None, calcType: CalcType = CalcType.TYPE_SUM):
        
        if grpLabel is not None:
            # Return a single value (Scalar)
            if calcType != CalcType.TYPE_COUNT_COIN_TYPE:
                return self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lLabelIdx[grpLabel]].agg(D_CALC_TYPE[calcType])
            elif calcType == CalcType.TYPE_COUNT_COIN_TYPE:
                return self.dfData[self.currencyColLabel][self.lLabelIdx[grpLabel]].count()

            
        else:
            # Work on all data, return a series
            ds_SentValue = pd.Series(index = self.dfData.index) #<! Maybe better just do dsTotalSentValue = self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]] ? Then data is local in cache
            if calcType != CalcType.TYPE_COUNT_COIN_TYPE:
                for ii in range(self.numGrps):
                    ds_SentValue[self.lLabelIdx[ii]] = self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lLabelIdx[ii]].agg(D_CALC_TYPE[calcType])
            
            elif calcType == CalcType.TYPE_COUNT_COIN_TYPE:
                for ii in range(self.numGrps):
                    ds_SentValue[self.lLabelIdx[ii]] = self.dfData[self.currencyColLabel][self.lLabelIdx[ii]].count()        
            
    
            return ds_SentValue


    @timer
    def _SentValue_time_diffs(self, amountCol: AmountType = AmountType.AMOUNT_USD, tokenId = None, grpLabel = None, calcType: CalcType = CalcType.TYPE_SUM):
        
        if grpLabel is not None:
            # Return a single value (Scalar)
            return self.dfData[self.timeColLabel][self.lLabelIdx[grpLabel]].diff().dt.total_seconds().agg(D_CALC_TYPE[calcType])
             
        else:
            # Work on all data, return a series
            ds_SentValue = pd.Series(index = self.dfData.index) #<! Maybe better just do dsTotalSentValue = self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]] ? Then data is local in cache
            
            for ii in range(self.numGrps):
                ds_SentValue[self.lLabelIdx[ii]] = self.dfData[self.timeColLabel][self.lLabelIdx[ii]].diff().dt.total_seconds().agg(D_CALC_TYPE[calcType])
                    
            return ds_SentValue

    @timer
    def _AnalyseRecieverId_new(self, amountCol = AmountType.AMOUNT_USD, calcType = CalcType.TYPE_SUM):
        
        
        ds_SentValue = pd.Series(index = self.dfData.index)
        for ii in range(len(self.lSubGrpUsrLabelIdx)):
            for i in range(len(self.lSubGrpUsrLabelIdx[ii])):
                ds_SentValue[self.lSubGrpUsrLabelIdx[ii][i][0]] =self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[ii][i][0]].agg(D_CALC_TYPE[calcType])            

        return ds_SentValue
 

    @timer
    def _AnalyseRecieverId_time_diffs(self, amountCol = AmountType.AMOUNT_USD, tokenId = None, grpLabel = None, subGrpLabel = None, calcType = CalcType.TYPE_SUM):

        ds_SentValue = pd.Series(index = self.dfData.index)
        for ii in range(len(self.lSubGrpUsrLabelIdx)):
            for i in range(len(self.lSubGrpUsrLabelIdx[ii])):
                ds_SentValue[self.lSubGrpUsrLabelIdx[ii][i][0]] =self.dfData[self.timeColLabel][self.lSubGrpUsrLabelIdx[ii][i][0]].diff().dt.total_seconds().agg(D_CALC_TYPE[calcType])

        return ds_SentValue 

    @timer
    @numba.jit
    def _SentValue_new_numba(self, amountCol: AmountType = AmountType.AMOUNT_USD, calcType: CalcType = CalcType.TYPE_SUM):
        
        ds_SentValue = pd.Series(index = self.dfData.index) 
        
        for ii in range(self.numGrps):
            #ds_SentValue[self.lLabelIdx[ii]] = self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lLabelIdx[ii]].agg(D_CALC_TYPE[calcType])
            ds_SentValue[self.lLabelIdx[ii]] = aggregator(self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lLabelIdx[ii]], D_CALC_TYPE[calcType])  
            
        return ds_SentValue

    @timer
    @numba.jit
    def _AnalyseRecieverId_new_numba(self, amountCol = AmountType.AMOUNT_USD, calcType = CalcType.TYPE_SUM):
        
        ds_SentValue = pd.Series(index = self.dfData.index)
        for ii in range(len(self.lSubGrpUsrLabelIdx)):
            for i in range(len(self.lSubGrpUsrLabelIdx[ii])):
                ds_SentValue[self.lSubGrpUsrLabelIdx[ii][i][0]] =self.dfData[D_AMOUNT_TYPE_COL_NAME[amountCol]][self.lSubGrpUsrLabelIdx[ii][i][0]].agg(D_CALC_TYPE[calcType])
                            

        return ds_SentValue

    @timer
    @numba.jit
    def _AnalyseRecieverId_time_diffs_numba(self, amountCol = AmountType.AMOUNT_USD, tokenId = None, grpLabel = None, subGrpLabel = None, calcType = CalcType.TYPE_SUM):
        
        ds_SentValue = pd.Series(index = self.dfData.index)
        for ii in range(len(self.lSubGrpUsrLabelIdx)):
            for i in range(len(self.lSubGrpUsrLabelIdx[ii])):
                ds_SentValue[self.lSubGrpUsrLabelIdx[ii][i][0]] =self.dfData[self.timeColLabel][self.lSubGrpUsrLabelIdx[ii][i][0]].diff().dt.total_seconds().agg(D_CALC_TYPE[calcType])

        return ds_SentValue


















