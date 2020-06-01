# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import glob
from os.path import join
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from numpy import unique
from time import time

dataRoot = '../../data/processed/'
datasetFilename = 'CICIDS2017_MasterData.csv'
datasetClean = 'CICIDS2017_CleanData.csv'

# display complete contents of a dataframe
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

# define undersampling dictionary
# used to establish the number of records
# for each class
DownSample = {
    'BENIGN' :                  25000, # we only want to downsample this to reduce bias
    'DoS Hulk' :                23012,
    'PortScan' :                15880,
    'DDoS' :                    12802,
    'DoS GoldenEye' :            10293,
    'FTP-Patator'   :             7938,
    'SSH-Patator'   :             5897,
    'DoS slowloris' :             5796,
    'DoS Slowhttptest':           5499,
    'Bot':                        1966,
    'Web Attack � Brute Force':   1507,
    'Web Attack � XSS' :           652,
    'Infiltration'  :                36,
    'Web Attack � Sql Injection':   21,
    'Heartbleed' :                   11
}

UpSample = {
    'BENIGN' :                  25000, # we only want to downsample this to reduce bias
    'DoS Hulk' :                23012,
    'PortScan' :                15880,
    'DDoS' :                    12802,
    'DoS GoldenEye' :            10293,
    'FTP-Patator'   :             7938,
    'SSH-Patator'   :             5897,
    'DoS slowloris' :             5796,
    'DoS Slowhttptest':           5499,
    'Bot':                        5000,
    'Web Attack � Brute Force':  5000,
    'Web Attack � XSS' :         5000,
    'Infiltration'  :             5000,
    'Web Attack � Sql Injection':5000,
    'Heartbleed' :                5000
}

def removeColumnsByVariance(X_train):
    # define the transform
    transform = VarianceThreshold()

    # transform the input data
    X_output = transform.fit_transform(X_train)

    return X_output    

def cleanData():

    t0 = time()    
    filepath = join(dataRoot,datasetFilename)
    df = pd.read_csv(filepath,low_memory=False)

    # convert inf to nan
    pd.set_option('mode.use_inf_as_na', True)

    # remove the spaces found in the dataset column names
    #
    df = df.rename(columns=lambda x: x.strip())

    df = df.dropna( axis=0, how='any')
    df.replace('Infinity',0.0, inplace=True)
    df = df.replace(',,', np.nan, inplace=False)
    df.replace('NaN',0.0, inplace=True)

    for i in df.columns:
        df = df[df[i] != "Infinity"]
    df = df[df[i] != np.nan]
    df = df[df[i] != ",,"]
    df[['Destination Port','Flow Bytes/s', 'Flow Packets/s']] = df[['Destination Port','Flow Bytes/s', 'Flow Packets/s']].apply(pd.to_numeric) 

    # check the data types
    if(not __debug__):
        print(df.dtypes)
        print(df.shape)

    df['Label'] = df['Label'].apply(lambda x: 0 if 'BENIGN' in x else 1)

    df = df.dropna()
    df.fillna(0, inplace=True)

    # summarize the number of unique values in each column
    unique_counts = df.nunique()

    # record columns to delete
    to_del = [i for i,v in enumerate(unique_counts) if v == 1]
    
    if(len(to_del) > 0):
        # drop useless columns
        df.drop(df.columns[to_del], axis=1, inplace=True)
    
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    print(X.shape, y.shape)

    X_new = removeColumnsByVariance(X)

    print(X_new.shape)

    #data.to_csv(datasetClean)

    tt = time() - t0
    print("Data Clean took {} seconds".format(round(tt,3)))
    return df

    """
    #print("Label Distribution:")
    #print(data['Label'].value_counts())

    nullColumns = data.columns[data.isnull().any()]

    if(len(nullColumns)):
        data = data.dropna(axis=1, how='any')

    if __debug__:
        print("NULL Columns: ")
        print(data[nullColumns].isnull().sum())

    #data=data.dropna( axis=0, how='any')
    #data=data.replace(',,', np.nan, inplace=False)
    #data.replace("Infinity", 0, inplace=True) 

    # determine if there are any NULL values
    nullValues = data[data.isnull().any(1)]
    
    if __debug__:
        print("NULL Values:")
        print(nullValues['Label'].value_counts())   

    # delete any record having a NULL value in the dataset
    #
    
    if(len(nullValues) > 0):
        print("Removing NULL values from dataset")
        data = data.dropna(axis=0, how='any')
    
     # remove any columns that could cause issues
    #dropColumns = ['Flow Packets/s','Flow Bytes/s']

    #data['Label'] = data['Label'].apply(lambda x: 0 if 'BENIGN' in x else 1)
    
    #data = data.drop(columns = dropColumns)
    #print("Dropping problematic columns: {}".format(dropColumns))

    #data.replace('Infinity',0.0, inplace=True)
    
    print("Replacing 'NaN' with 0's...")
    data.replace('NaN',0.0, inplace=True)

    #tmpdata = data.copy()
    #tmpdata = tmpdata.drop(columns = 'Label')
    #constant_features = [feat for feat in tmpdata.columns if data[feat].std() == 0]

    # dropping bad columns, they coause issues due to large numbers
    #data = data.drop(columns=constant_features)
    #print("Dropped the following constant features: {}".format(','.join(constant_features)))

    # fill any columns w/ missing features with a mean value
    #data.fillna(data.mean(), inplace = True) 

    # lets count if there is NaN values in our dataframe( AKA missing features)
    assert data.isnull().sum().sum()==0, "There should not be any NaN"

    #print("Completed dropping columns: {}".format(dropColumns))
    print("Completed cleaning dataset")
    print("Finished Reading Data File: {}...".format(datasetFilename))

    print(len(data.columns))

    return data
    """

def scaleData(data):
    
    print("Running Scale Data...")
    # Initialize our scalar
    scaler = StandardScaler() 

    # apply the scalar
    scaler.fit_transform(data) 
    
    print("Finished Running Scale Data...")

    return data


def balanceData(attacks, y):
    downSample = RandomUnderSampler(sampling_strategy=DownSample, random_state=0)
    upSample = RandomOverSampler(sampling_strategy=UpSample, random_state=0)

    downSampleAttacks, downSampleAttacksLabel = downSample.fit_sample(attacks,y)
    print(attacks.describe())
    print(downSampleAttacksLabel.describe())

    #upSampleAttacks, upSampleAttacksLabel = upSample.fit_sample(downSampleAttacks,downSampleAttacksLabel)
    #print(upSampleAttacksLabel.describe())
    
    #print(upSampleAttacks.describe())

    #return upSampleAttacks, upSampleAttacksLabel

def load_data(dataroot):
    
    print("Start Loading Data...")

    data = cleanData()
    
    #train_X = data.drop(columns=['Label']).copy()
    #train_y = data[['Label']].copy()
    #print(train_X.head())
    #print(train_y['Label'].value_counts())
    #labelencoder = LabelEncoder()
    #X = data.iloc[:, :-1]
    #y = data.iloc[:,-1]
    
    
    #y = labelencoder.fit_transform(y)
    #y = data.iloc[:,-1]
    #y = labelencoder.fit_transform(y)
    #X_train, y_train = balanceData(train_X,train_y)
    #balanceData(train_X,train_y)
    #X = data.iloc[:,0:len(data.columns)-1].abs()
    #y = data.iloc[:,-1]
    
    #X = data.iloc[:,0:len(data.columns)-1].abs()
    #y = data.iloc[:,-1]
    
    #X = scaleData(X)
    #print(data['Label'].value_counts())

    #r,c = data.shape
    #print("There are {} flow records with {} feature dimension".format(r, c))

    #X_train['Label']
    #X_train.to_csv('CICIDS2017_cleaned.csv')


def main():
       
    dataroot = '../../data/raw/'
    load_data(dataroot)
    

if __name__ == "__main__":
    main()
