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

from DataSetsAuxFun import *

# Typing
from typing import Dict, List


#TODO: Change names to match the feature calculation

@unique
class FeatureName(Enum): #<! The list of defined features
    AMOUNT_SUM_ASSET                    = auto()
    AMOUNT_SUM_USR                      = auto()
    AMOUNT_MEAN_ASSET                   = auto()
    AMOUNT_MEAN_USR                     = auto()
    AMOUNT_STD_ASSET                    = auto()
    AMOUNT_STD_USR                      = auto()
    AMOUNT_VAR_ASSET                    = auto()
    AMOUNT_VAR_USR                      = auto()
    AMOUNT_MEDIAN_ASSET                 = auto()
    AMOUNT_MEDIAN_USR                   = auto()
    AMOUNT_MIN_ASSET                    = auto()
    AMOUNT_MIN_USR                      = auto()
    AMOUNT_MAX_ASSET                    = auto()
    AMOUNT_MAX_USR                      = auto()
    # AMOUNT_PCTILE_ASSET                 = auto() #<! Quantile
    # AMOUNT_PCTILE_USR                   = auto() #<! Quantile
    TIME_DIFF_MEAN_ASSET                = auto()
    TIME_DIFF_MEAN_USR                  = auto()
    TIME_DIFF_STD_ASSET                 = auto()
    TIME_DIFF_STD_USR                   = auto()
    TIME_DIFF_MEDIAN_ASSET              = auto()
    TIME_DIFF_MEDIAN_USR                = auto()
    TIME_DIFF_MIN_ASSET                 = auto()
    TIME_DIFF_MIN_USR                   = auto()
    TIME_DIFF_MAX_ASSET                 = auto()
    TIME_DIFF_MAX_USR                   = auto()
    COIN_TYPE_COUNT_ASSET               = auto()
    COIN_TYPE_COUNT_USR_MEAN_ASSET      = auto()  # <<--  The average of the number of types of all user for an asset
    COIN_TYPE_USR_MEAN_ASSET_RATIO      = auto()  # <<--  The ratio between a specific user to the average of the asset
    COIN_TYPE_COUNT_USR                 = auto()
    RECEIVER_TYPE_COUNT_ASSET           = auto()
    RECEIVER_TYPE_COUNT_USR             = auto()
    # COIN_TYPE_COUNT_USR_MEAN            = auto()
    TIME_HOUR                           = auto()
    TIME_WEEKDAY                        = auto()
    TIME_MIN                            = auto()
    TIME_MAX                            = auto()
    TIME_INTERVL_USR                    = auto()
    AMOUNT_STD_RATIO_USR_ASSET          = auto()
    AMOUNT_MEAN_RATIO_USR_ASSET         = auto()
    TIME_DIFF_STD_RATIO_USR_ASSET       = auto()
    TIME_DIFF_MEAN_RATIO_USR_ASSET      = auto()
    TSX_COUNT_ASSET                     = auto()
    TSX_COUNT_USR                       = auto()
    TSX_FREQ_HZ_USR                     = auto()
    GAS_PRICE_MEAN_ASSET                = auto()
    GAS_PRICE_MEAN_USR                  = auto()
    GAS_PRICE_STD_ASSET                 = auto()
    GAS_PRICE_STD_USR                   = auto()
    GAS_PRICE_MEDIAN_ASSET              = auto()
    GAS_PRICE_MEDIAN_USR                = auto()
    GAS_LIMIT_MEAN_ASSET                = auto()
    GAS_LIMIT_MEAN_USR                  = auto()
    GAS_LIMIT_STD_ASSET                 = auto()
    GAS_LIMIT_STD_USR                   = auto()
    GAS_LIMIT_MEDIAN_ASSET              = auto()
    GAS_LIMIT_MEDIAN_USR                = auto()
    GAS_USED_MEAN_ASSET                 = auto()
    GAS_USED_MEAN_USR                   = auto()
    GAS_USED_STD_ASSET                  = auto()
    GAS_USED_STD_USR                    = auto()
    GAS_USED_MEDIAN_ASSET               = auto()
    GAS_USED_MEDIAN_USR                 = auto()
    MIN_INDICATOR                       = auto()
    GAS_PRICE_USR_ASSET_RATIO_MEAN      = auto()
    GAS_LIMIT_USR_ASSET_RATIO_MEAN      = auto()
    GAS_USED_USR_ASSET_RATIO_MEAN       = auto()
    GAS_PRICE_LIMIT_RATIO               = auto()
    GAS_PRICE_USED_RATIO                = auto()
    GAS_USED_LIMIT_RATIO                = auto()
    GAS_PRICE_LIMIT_RATIO_MEAN          = auto()
    GAS_PRICE_USED_RATIO_MEAN           = auto()
    GAS_USED_LIMIT_RATIO_MEAN           = auto()
    GAS_PRICE_QUANTILE_RATIO            = auto()
    GAS_USED_QUANTILE_RATIO             = auto()
    GAS_LIMIT_QUANTILE_RATIO            = auto()
    GAS_PRICE_QUANTILE_USR              = auto()
    GAS_USED_QUANTILE_USR               = auto()
    GAS_LIMIT_QUANTILE_USR              = auto()


# Maps a feature to a function



# def ApplyListOfFeatures(dfData: pd.DataFrame, feature_list: List, dFeatureFunMap: Dict = dFeatureFunMap): #<! Royi: @Anton, why do you use both dfGbs and df? 
def ApplyListOfFeatures(dfData: pd.DataFrame, feature_list: List): #<! Royi: @Anton, why do you use both dfGbs and df? 

    dfGbs = dfData.GrpBySender
    
    dFeatureFunMap = {  FeatureName.AMOUNT_SUM_ASSET.name           : {'func' : dfGbs.AggBySender,   'col': dfGbs.amountUSDColLabel,      'method' : CalcType.TYPE_SUM } ,  
                            FeatureName.AMOUNT_MEAN_ASSET.name          : {'func' : dfGbs.AggBySender,   'col': dfGbs.amountUSDColLabel,      'method' : CalcType.TYPE_MEAN } ,
                            FeatureName.AMOUNT_STD_ASSET.name           : {'func' : dfGbs.AggBySender,   'col': dfGbs.amountUSDColLabel,      'method' : CalcType.TYPE_STD } ,
                            FeatureName.AMOUNT_VAR_ASSET.name           : {'func' : dfGbs.AggBySender,   'col': dfGbs.amountUSDColLabel,      'method' : CalcType.TYPE_VAR } ,  
                            FeatureName.AMOUNT_MEDIAN_ASSET.name        : {'func' : dfGbs.AggBySender,   'col': dfGbs.amountUSDColLabel,      'method' : CalcType.TYPE_MEDIAN } , 
                            FeatureName.AMOUNT_MIN_ASSET.name           : {'func' : dfGbs.AggBySender,   'col': dfGbs.amountUSDColLabel,      'method' : CalcType.TYPE_MIN } ,
                            FeatureName.AMOUNT_MAX_ASSET.name           : {'func' : dfGbs.AggBySender,   'col': dfGbs.amountUSDColLabel,      'method' : CalcType.TYPE_MAX } ,
                            FeatureName.TSX_COUNT_ASSET.name            : {'func' : dfGbs.AggBySender,   'col': dfGbs.amountUSDColLabel,      'method' : CalcType.TYPE_COUNT } ,
                            FeatureName.COIN_TYPE_COUNT_ASSET.name      : {'func' : dfGbs.AggBySender,   'col': dfGbs.currencyColLabel,       'method' : CalcType.TYPE_COUNT_COIN_TYPE } ,
                            FeatureName.RECEIVER_TYPE_COUNT_ASSET.name  : {'func' : dfGbs.AggBySender,   'col': dfGbs.receiverTypeColLabel,   'method' : CalcType.TYPE_COUNT_RECEIVER_TYPE } ,
                            FeatureName.GAS_PRICE_MEAN_ASSET.name       : {'func' : dfGbs.AggBySender,   'col': dfGbs.gasPriceColLabel,       'method' : CalcType.TYPE_MEAN },
                            FeatureName.GAS_PRICE_STD_ASSET.name        : {'func' : dfGbs.AggBySender,   'col': dfGbs.gasPriceColLabel,       'method' : CalcType.TYPE_STD },   
                            FeatureName.GAS_PRICE_MEDIAN_ASSET.name     : {'func' : dfGbs.AggBySender,   'col': dfGbs.gasPriceColLabel,       'method' : CalcType.TYPE_MEDIAN },  
                            FeatureName.GAS_LIMIT_MEAN_ASSET.name       : {'func' : dfGbs.AggBySender,   'col': dfGbs.gasLimitColLabel,       'method' : CalcType.TYPE_MEAN },
                            FeatureName.GAS_LIMIT_STD_ASSET.name        : {'func' : dfGbs.AggBySender,   'col': dfGbs.gasLimitColLabel,       'method' : CalcType.TYPE_STD },
                            FeatureName.GAS_LIMIT_MEDIAN_ASSET.name     : {'func' : dfGbs.AggBySender,   'col': dfGbs.gasLimitColLabel,       'method' : CalcType.TYPE_MEDIAN },
                            FeatureName.GAS_USED_MEAN_ASSET.name        : {'func' : dfGbs.AggBySender,   'col': dfGbs.gasUsedColLabel,        'method' : CalcType.TYPE_MEAN },
                            FeatureName.GAS_USED_STD_ASSET.name         : {'func' : dfGbs.AggBySender,   'col': dfGbs.gasUsedColLabel,        'method' : CalcType.TYPE_STD }, 
                            FeatureName.GAS_USED_MEDIAN_ASSET.name      : {'func' : dfGbs.AggBySender,   'col': dfGbs.gasUsedColLabel,        'method' : CalcType.TYPE_MEDIAN },
                            FeatureName.TIME_DIFF_MEAN_ASSET.name       : {'func' : dfGbs.AggBySender,   'col': dfGbs.timeDiffAssetColLabel,  'method' : CalcType.TYPE_TIME_DIFF_MEAN } ,
                            FeatureName.TIME_DIFF_STD_ASSET.name        : {'func' : dfGbs.AggBySender,   'col': dfGbs.timeDiffAssetColLabel,  'method' : CalcType.TYPE_TIME_DIFF_STD } ,
                            FeatureName.TIME_DIFF_MEDIAN_ASSET.name     : {'func' : dfGbs.AggBySender,   'col': dfGbs.timeDiffAssetColLabel,  'method' : CalcType.TYPE_TIME_DIFF_MEDIAN } ,
                            FeatureName.TIME_DIFF_MIN_ASSET.name        : {'func' : dfGbs.AggBySender,   'col': dfGbs.timeDiffAssetColLabel,  'method' : CalcType.TYPE_TIME_DIFF_MIN } , 
                            FeatureName.TIME_DIFF_MAX_ASSET.name        : {'func' : dfGbs.AggBySender,   'col': dfGbs.timeDiffAssetColLabel,  'method' : CalcType.TYPE_TIME_DIFF_MAX } ,
                            FeatureName.AMOUNT_SUM_USR.name             : {'func' : dfGbs.AggByReceiver, 'col': dfGbs.amountUSDColLabel,      'method' : CalcType.TYPE_SUM }, 
                            FeatureName.AMOUNT_MEAN_USR.name            : {'func' : dfGbs.AggByReceiver, 'col': dfGbs.amountUSDColLabel,      'method' : CalcType.TYPE_MEAN } ,
                            FeatureName.AMOUNT_STD_USR.name             : {'func' : dfGbs.AggByReceiver, 'col': dfGbs.amountUSDColLabel,      'method' : CalcType.TYPE_STD } ,
                            FeatureName.AMOUNT_VAR_USR.name             : {'func' : dfGbs.AggByReceiver, 'col': dfGbs.amountUSDColLabel,      'method' : CalcType.TYPE_VAR }, 
                            FeatureName.AMOUNT_MEDIAN_USR.name          : {'func' : dfGbs.AggByReceiver, 'col': dfGbs.amountUSDColLabel,      'method' : CalcType.TYPE_MEDIAN },
                            FeatureName.AMOUNT_MIN_USR.name             : {'func' : dfGbs.AggByReceiver, 'col': dfGbs.amountUSDColLabel,      'method' : CalcType.TYPE_MIN },
                            FeatureName.AMOUNT_MAX_USR.name             : {'func' : dfGbs.AggByReceiver, 'col': dfGbs.amountUSDColLabel,      'method' : CalcType.TYPE_MAX },
                            FeatureName.TSX_COUNT_USR.name              : {'func' : dfGbs.AggByReceiver, 'col': dfGbs.amountUSDColLabel,      'method' : CalcType.TYPE_COUNT },
                            FeatureName.COIN_TYPE_COUNT_USR.name        : {'func' : dfGbs.AggByReceiver, 'col': dfGbs.currencyColLabel,       'method' : CalcType.TYPE_COUNT_COIN_TYPE }, 
                            FeatureName.RECEIVER_TYPE_COUNT_USR.name    : {'func' : dfGbs.AggByReceiver, 'col': dfGbs.receiverTypeColLabel,   'method' : CalcType.TYPE_COUNT_RECEIVER_TYPE } ,
                            FeatureName.GAS_PRICE_MEAN_USR.name         : {'func' : dfGbs.AggByReceiver, 'col': dfGbs.gasPriceColLabel,       'method' : CalcType.TYPE_MEAN } ,
                            FeatureName.GAS_PRICE_STD_USR.name          : {'func' : dfGbs.AggByReceiver, 'col': dfGbs.gasPriceColLabel,       'method' : CalcType.TYPE_STD },
                            FeatureName.GAS_PRICE_MEDIAN_USR.name       : {'func' : dfGbs.AggByReceiver, 'col': dfGbs.gasPriceColLabel,       'method' : CalcType.TYPE_MEDIAN },
                            FeatureName.GAS_LIMIT_MEAN_USR.name         : {'func' : dfGbs.AggByReceiver, 'col': dfGbs.gasLimitColLabel,       'method' : CalcType.TYPE_MEAN },
                            FeatureName.GAS_LIMIT_STD_USR.name          : {'func' : dfGbs.AggByReceiver, 'col': dfGbs.gasLimitColLabel,       'method' : CalcType.TYPE_STD },
                            FeatureName.GAS_LIMIT_MEDIAN_USR.name       : {'func' : dfGbs.AggByReceiver, 'col': dfGbs.gasLimitColLabel,       'method' : CalcType.TYPE_MEDIAN },
                            FeatureName.GAS_USED_MEAN_USR.name          : {'func' : dfGbs.AggByReceiver, 'col': dfGbs.gasUsedColLabel,        'method' : CalcType.TYPE_MEAN } ,
                            FeatureName.GAS_USED_STD_USR.name           : {'func' : dfGbs.AggByReceiver, 'col': dfGbs.gasUsedColLabel,        'method' : CalcType.TYPE_STD },
                            FeatureName.GAS_USED_MEDIAN_USR.name        : {'func' : dfGbs.AggByReceiver, 'col': dfGbs.gasUsedColLabel,        'method' : CalcType.TYPE_MEDIAN },
                            FeatureName.TIME_DIFF_MEAN_USR.name         : {'func' : dfGbs.AggByReceiver, 'col': dfGbs.timeDiffUserColLabel,   'method' : CalcType.TYPE_TIME_DIFF_MEAN } ,
                            FeatureName.TIME_DIFF_STD_USR.name          : {'func' : dfGbs.AggByReceiver, 'col': dfGbs.timeDiffUserColLabel,   'method' : CalcType.TYPE_TIME_DIFF_STD },
                            FeatureName.TIME_DIFF_MEDIAN_USR.name       : {'func' : dfGbs.AggByReceiver, 'col': dfGbs.timeDiffUserColLabel,   'method' : CalcType.TYPE_TIME_DIFF_MEDIAN }, 
                            FeatureName.TIME_DIFF_MIN_USR.name          : {'func' : dfGbs.AggByReceiver, 'col': dfGbs.timeDiffUserColLabel,   'method' : CalcType.TYPE_TIME_DIFF_MIN },
                            FeatureName.TIME_DIFF_MAX_USR.name          : {'func' : dfGbs.AggByReceiver, 'col': dfGbs.timeDiffUserColLabel,   'method' : CalcType.TYPE_TIME_DIFF_MAX},
                            FeatureName.TIME_HOUR.name                  : {'func' : dfGbs.GetTimeVals,   'col': '',                           'param'  : PeriodTimeType.HOUR_DAY},  
                            FeatureName.TIME_WEEKDAY.name               : {'func' : dfGbs.GetTimeVals,   'col': '',                           'param'  : PeriodTimeType.DAY_WEEK},
                            FeatureName.TIME_MAX.name                   : {'func' : dfGbs.AggByReceiver, 'col': dfGbs.timeColLabel,            'method' : CalcType.TYPE_MAX },
                            FeatureName.TIME_MIN.name                   : {'func' : dfGbs.AggByReceiver, 'col': dfGbs.timeColLabel,            'method' : CalcType.TYPE_MIN },
                            FeatureName.GAS_PRICE_QUANTILE_USR.name     : {'func' : dfGbs.dfSubGrpByRec[dfGbs.gasPriceColLabel].transform('quantile' ,q = 0.75)},
                            FeatureName.GAS_LIMIT_QUANTILE_USR.name     : {'func' : dfGbs.dfSubGrpByRec[dfGbs.gasLimitColLabel].transform('quantile' ,q = 0.75)},
                            FeatureName.GAS_USED_QUANTILE_USR.name      : {'func' : dfGbs.dfSubGrpByRec[dfGbs.gasUsedColLabel].transform('quantile' ,q = 0.75)}
                                                
                                 }
                    
    for feat in feature_list:
        if   feat in [FeatureName.TIME_HOUR.name, FeatureName.TIME_WEEKDAY.name]:
            dfData[feat] = dFeatureFunMap[feat]['func'](periodTimeType = dFeatureFunMap[feat]['param'])
        elif feat in [FeatureName.AMOUNT_MEAN_RATIO_USR_ASSET.name, 
                      FeatureName.AMOUNT_STD_RATIO_USR_ASSET.name, 
                      FeatureName.TIME_DIFF_MEAN_RATIO_USR_ASSET.name, 
                      FeatureName.TIME_DIFF_STD_RATIO_USR_ASSET.name]:
            
            if feat == FeatureName.AMOUNT_MEAN_RATIO_USR_ASSET.name:
                f1 = FeatureName.AMOUNT_MEAN_USR.name ; f2 = FeatureName.AMOUNT_MEAN_ASSET.name
                
            if feat == FeatureName.AMOUNT_STD_RATIO_USR_ASSET.name:
                f1 = FeatureName.AMOUNT_STD_USR.name ; f2 = FeatureName.AMOUNT_STD_ASSET.name
    
            if feat == FeatureName.TIME_DIFF_MEAN_RATIO_USR_ASSET.name:
                f1 = FeatureName.TIME_DIFF_MEAN_USR.name ; f2 = FeatureName.TIME_DIFF_MEAN_ASSET.name

            if feat == FeatureName.TIME_DIFF_STD_RATIO_USR_ASSET.name:
                f1 = FeatureName.TIME_DIFF_STD_USR.name ; f2 = FeatureName.TIME_DIFF_STD_ASSET.name
            
            nom = dFeatureFunMap[f1]['func'](colName = dFeatureFunMap[f1]['col'], grpLabel = None, calcType = dFeatureFunMap[f1]['method'])  
            denom = dFeatureFunMap[f2]['func'](colName = dFeatureFunMap[f2]['col'], grpLabel = None, calcType = dFeatureFunMap[f2]['method'])
            dfData[feat] = nom/denom  
        
        elif feat in [FeatureName.TIME_INTERVL_USR.name,
                      FeatureName.TSX_FREQ_HZ_USR.name]:
            max = dfGbs.AggByReceiver(colName = dfGbs.timeColLabel, grpLabel = None, subGrpLabel = None, calcType = CalcType.TYPE_MAX)
            min = dfGbs.AggByReceiver(colName = dfGbs.timeColLabel, grpLabel = None, subGrpLabel = None, calcType = CalcType.TYPE_MIN)
            
            if feat == FeatureName.TIME_INTERVL_USR.name:
                dfData[feat] = (max - min).dt.total_seconds()
            if feat == FeatureName.TSX_FREQ_HZ_USR.name:
                dfData[feat] = dfGbs.AggByReceiver(colName = dfGbs.amountUSDColLabel, grpLabel = None, subGrpLabel = None, calcType = CalcType.TYPE_COUNT) / (max - min).dt.total_seconds()
        
        elif feat in [FeatureName.GAS_PRICE_USR_ASSET_RATIO_MEAN.name,
                      FeatureName.GAS_LIMIT_USR_ASSET_RATIO_MEAN.name,
                      FeatureName.GAS_USED_USR_ASSET_RATIO_MEAN.name,
                      FeatureName.GAS_PRICE_LIMIT_RATIO.name,
                      FeatureName.GAS_PRICE_USED_RATIO.name,
                      FeatureName.GAS_USED_LIMIT_RATIO.name,              
                      FeatureName.GAS_PRICE_LIMIT_RATIO_MEAN.name,
                      FeatureName.GAS_PRICE_USED_RATIO_MEAN.name,
                      FeatureName.GAS_USED_LIMIT_RATIO_MEAN.name,
                      FeatureName.GAS_PRICE_QUANTILE_RATIO.name ,
                      FeatureName.GAS_LIMIT_QUANTILE_RATIO.name ,
                      FeatureName.GAS_USED_QUANTILE_RATIO.name]:


            if feat == FeatureName.GAS_PRICE_USR_ASSET_RATIO_MEAN.name:
                nom = dfGbs.AggByReceiver(colName = dfGbs.gasPriceColLabel, grpLabel = None, subGrpLabel = None, calcType = CalcType.TYPE_MEAN)  
                denom = dfGbs.AggBySender(colName = dfGbs.gasPriceColLabel, grpLabel = None, calcType = CalcType.TYPE_MEAN) 
                       
                
            if feat == FeatureName.GAS_LIMIT_USR_ASSET_RATIO_MEAN.name:
                nom = dfGbs.AggByReceiver(colName = dfGbs.gasLimitColLabel, grpLabel = None, subGrpLabel = None, calcType = CalcType.TYPE_MEAN)  
                denom = dfGbs.AggBySender(colName = dfGbs.gasLimitColLabel, grpLabel = None, calcType = CalcType.TYPE_MEAN)
                
            
            if feat == FeatureName.GAS_USED_USR_ASSET_RATIO_MEAN.name:
                nom = dfGbs.AggByReceiver(colName = dfGbs.gasUsedColLabel, grpLabel = None, subGrpLabel = None, calcType = CalcType.TYPE_MEAN)  
                denom = dfGbs.AggBySender(colName = dfGbs.gasUsedColLabel, grpLabel = None, calcType = CalcType.TYPE_MEAN)            

            if feat == FeatureName.GAS_PRICE_LIMIT_RATIO.name:
                nom = dfGbs.dfData['Gas Price']   
                denom = dfGbs.dfData['Gas Limit'] 
                       
            if feat == FeatureName.GAS_PRICE_USED_RATIO.name:
                nom = dfGbs.dfData['Gas Price']
                denom = dfGbs.dfData['Gas Used']
            
            if feat == FeatureName.GAS_USED_LIMIT_RATIO.name:
                nom = dfGbs.dfData['Gas Used']
                denom = dfGbs.dfData['Gas Limit']
        

            if feat == FeatureName.GAS_PRICE_LIMIT_RATIO_MEAN.name:
                nom = dfGbs.AggByReceiver(colName = dfGbs.gasPriceColLabel, grpLabel = None, subGrpLabel = None, calcType = CalcType.TYPE_MEAN)  
                denom = dfGbs.AggByReceiver(colName = dfGbs.gasLimitColLabel, grpLabel = None, subGrpLabel = None, calcType = CalcType.TYPE_MEAN) 
                       
            if feat == FeatureName.GAS_PRICE_USED_RATIO_MEAN.name:
                nom = dfGbs.AggByReceiver(colName = dfGbs.gasPriceColLabel, grpLabel = None, subGrpLabel = None, calcType = CalcType.TYPE_MEAN)  
                denom = dfGbs.AggByReceiver(colName = dfGbs.gasUsedColLabel, grpLabel = None, subGrpLabel = None, calcType = CalcType.TYPE_MEAN)
                
            
            if feat == FeatureName.GAS_USED_LIMIT_RATIO_MEAN.name:
                nom = dfGbs.AggByReceiver(colName = dfGbs.gasUsedColLabel, grpLabel = None, subGrpLabel = None, calcType = CalcType.TYPE_MEAN)  
                denom = dfGbs.AggByReceiver(colName = dfGbs.gasPriceColLabel, grpLabel = None, subGrpLabel = None, calcType = CalcType.TYPE_MEAN)
                
 
            if feat == FeatureName.GAS_PRICE_QUANTILE_RATIO.name:
                nom = dfGbs.dfData['Gas Price']  
                denom =dfGbs.dfSubGrpByRec[dfGbs.gasPriceColLabel].transform('quantile' ,q =0.75) 
                       
            if feat == FeatureName.GAS_LIMIT_QUANTILE_RATIO.name:
                nom = dfGbs.dfData['Gas Limit']  
                denom =dfGbs.dfSubGrpByRec[dfGbs.gasLimitColLabel].transform('quantile' ,q =0.75)
                
            
            if feat == FeatureName.GAS_USED_QUANTILE_RATIO.name:
                nom = dfGbs.dfData['Gas Used']  
                denom =dfGbs.dfSubGrpByRec[dfGbs.gasUsedColLabel].transform('quantile' ,q =0.75)
            
            dfData[feat] = nom/denom
        
        elif feat == FeatureName.MIN_INDICATOR.name:
            feat_ = FeatureName.TIME_MIN.name
            
            if feat_ not in dfData.columns:
                dfData[feat_] = dFeatureFunMap[feat_]['func'](colName = dFeatureFunMap[feat_]['col'], grpLabel = None, calcType = dFeatureFunMap[feat_]['method'])
                
            dfData[feat] = 0 ; dfData.loc[dfData[dfGbs.timeColLabel] == dfData[feat_], feat] = 1   
            

        elif feat == FeatureName.COIN_TYPE_COUNT_USR_MEAN_ASSET.name:
            dfData[feat]    = dfGbs.AvgByUserCoinType()
        
        elif feat == FeatureName.COIN_TYPE_USR_MEAN_ASSET_RATIO.name:
            dfData[feat]  = dfGbs.AggByReceiver(colName = dfGbs.currencyColLabel, grpLabel = None, calcType = CalcType.TYPE_COUNT_COIN_TYPE) / dfGbs.AvgByUserCoinType()
 

        else:
            if 'method' in dFeatureFunMap[feat]:
                if feat !=FeatureName.TIME_MIN.name:
                    dfData[feat] = dFeatureFunMap[feat]['func'](colName = dFeatureFunMap[feat]['col'], grpLabel = None, calcType = dFeatureFunMap[feat]['method'])
            elif 'method' not in dFeatureFunMap[feat]:
                dfData[feat] = dFeatureFunMap[feat]['func']      
        

    return dfData
    