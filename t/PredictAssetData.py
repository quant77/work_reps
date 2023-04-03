'''
Class to initiate the model by a path to a model file.
The constructor will have the following parameters:
modelPath - Path to a model to load.
lFeaturesRaw - A list of the RAW features.
lFeaturesProcess - A list of features to process.
It will have the methods:
ValidateData()
Input: The DF of RAW data.
Output: True / False (Data is valid).
Inside it initiates our own Pandas Extension.
CalculateFeatures()
Input: The DF of RAW data.
Output: Updated DF with the algorithm features.
PredictLabels()
Input: The processed DF.
Output: Processed DF with updated labels.
The file will also include a function to pre process and validate the data (Should be called prior to the use of the class).
'''

# General Tools

# Misc
import datetime
from platform import python_version
import random
import warnings

# Ensemble Engines
from xgboost import XGBClassifier

# Visualization
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from DataSetsAuxFun import *

from joblib import load
import pickle

class PredictAssetData():
    def __init__(self, 
                 modelFolderPath : str,
                 modelFileName: str = 'Model.pkl', 
                 lRawFeaturesFileName: str = 'lRawFeatures.pkl', 
                 lProcessedFeatures: str = 'lProcessedFeatures.pkl', 
                 lSelectedFeatures: str = 'lSelectedFeatures.pkl',
                 scaler_dct: str =  'scaler_dct.pkl',
                 lCatFeatures: str = 'lCatFeatures.pkl'):
        
        self.modelFolderPath        =  modelFolderPath
        self.predictedLabelsColName = 'Label_predicted'

        self.lRawFeatures       = pickle.load(open(os.path.join(self.modelFolderPath, lRawFeaturesFileName), 'rb'))
        self.lProcessedFeatures = pickle.load(open(os.path.join(self.modelFolderPath, lProcessedFeatures), 'rb'))
        self.lSelectedFeatures  = pickle.load(open(os.path.join(self.modelFolderPath, lSelectedFeatures), 'rb'))
        self.lCatFeatures       = pickle.load(open(os.path.join(self.modelFolderPath, lCatFeatures), 'rb'))
        self.scaler_dct       = pickle.load(open(os.path.join(self.modelFolderPath, scaler_dct), 'rb'))
        self.model              = pickle.load(open(os.path.join(self.modelFolderPath, modelFileName), 'rb'))
        
    def ValidateData(self, dfData: pd.DataFrame) -> pd.DataFrame:
        
        lDfColName = dfData.columns.to_list()

        for colName in self.lRawFeatures:
            if colName not in lDfColName:
                raise ValueError(f'The column {colName} is missing in the data frame')
        
        # Initialize the Pandas Extension
        numGrps = dfData.GrpBySender.numGrps
        #if numGrps > 1:
        #    raise ValueError(f'The data includes {numGrps} assets instead of 1')

        return dfData

    def CalculateFeatures(self, dfData: pd.DataFrame ) -> pd.DataFrame:
        '''
        Input: The DF of RAW data.
        Output: Updated DF with the algorithm features.
        '''
        #### basically use my function to apply feature generation using some way
        
        dfData = ApplyListOfFeatures(dfData, self.lProcessedFeatures)

        return dfData
    
    def GenDataPredict(self, dfData: pd.DataFrame ) -> pd.DataFrame:
        '''
        Input: The DF of RAW data.
        Output: Updated DF with the algorithm features.
        '''

        dfX = dfData[self.lSelectedFeatures].copy()
        dfX.replace([np.inf, -np.inf], np.nan, inplace = True)
        dfX.fillna(0, inplace = True)
        dfX.rename(columns = {'Amount [USD]': 'Amount USD'}, inplace = True)

        #hStdScaler = StandardScaler()
        #dfX[self.lSelectedFeatures] = hStdScaler.fit_transform(dfX[self.lSelectedFeatures])
        for f in self.lSelectedFeatures:
           hStdScaler = self.scaler_dct[f]
           dfX[f] = hStdScaler.transform(dfX[[f]])



        for catColName in self.lCatFeatures:
            if catColName in dfX.columns:
                dfX[catColName] = dfX[catColName].astype('category', copy = False)

        return dfX

        
        
        

        


    def PredictLabels(self, dfX: pd.DataFrame, dfData: pd.DataFrame) -> pd.DataFrame:
       '''
       PredictLabels()
       Input: The processed DF.
       Output: Processed DF with updated labels.
       ''' 
       ###### add function to train model
       
        
       predictedLabels = self.model.predict(dfX)
       dfData[self.predictedLabelsColName] = predictedLabels  
 
       return dfData   