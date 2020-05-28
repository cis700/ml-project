import glob
from os.path import join
import pandas as pd
#import dask.dataframe as dd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import pickle

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

def readData(dataroot,filename, datasetName):

    filepath = join(dataroot,filename)
    data = pd.read_csv(filepath)

    # remove the spaces found in the dataset column names
    #
    data = data.rename(columns=lambda x: x.strip())

    #target = data['Label'].apply(lambda x: 0 if 'BENIGN' in x else 1)
    
    print("Label Distribution for {}:".format(datasetName))
    print(data['Label'].value_counts())

    nullColumns = data.columns[data.isnull().any()]
    #print(data[nullColumns].isnull().sum())

    data=data.dropna( axis=0, how='any')
    data=data.replace(',,', np.nan, inplace=False)
    data.replace("Infinity", 0, inplace=True) 



    #pd.set_option('display.max_rows', data.shape[0])

    # determine 
    nullValues = data[data.isnull().any(1)]
    print(nullValues['Label'].value_counts())

    if(len(nullColumns)):
        data = data.dropna(axis=1, how='any')

    # delete any record having a NULL value in the dataset
    #
    if(len(nullValues) > 0):
        print("Cleaning NULL values from dataset '{}'".format(datasetName))
        data = data.dropna(axis=0, how='any')

     # remove any columns that could cause issues
    #dropColumns = ['Flow Packets/s','Flow Bytes/s']

    #data['Label'] = data['Label'].apply(lambda x: 0 if 'BENIGN' in x else 1)
    
    #data = data.drop(columns = dropColumns)
    #print("Dropping problematic columns: {}".format(dropColumns))

    data.replace('Infinity',0.0, inplace=True)
    data.replace('NaN',0.0, inplace=True)

    #tmpdata = data.copy()
    #tmpdata = tmpdata.drop(columns = 'Label')
    #constant_features = [feat for feat in tmpdata.columns if data[feat].std() == 0]

    # dropping bad columns, they coause issues due to large numbers
    #data = data.drop(columns=constant_features)
    #print("Dropped the following constant features: {}".format(','.join(constant_features)))

    # fill any columns w/ missing features with a mean value
    data.fillna(data.mean(), inplace = True) 

    # lets count if there is NaN values in our dataframe( AKA missing features)
    assert data.isnull().sum().sum()==0, "There should not be any NaN"

    #print("Completed dropping columns: {}".format(dropColumns))
    print("Completed cleaning of dataset '{}'".format(datasetName))
    print("Finished Reading Data File '{}'...".format(filename))

    print(len(data.columns))

    return data

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

    upSampleAttacks, upSampleAttacksLabel = upSample.fit_sample(downSampleAttacks,downSampleAttacksLabel)
    print(upSampleAttacksLabel.describe())
    
    print(upSampleAttacks.describe())

    return upSampleAttacks, upSampleAttacksLabel

def load_data(dataroot):
    
    print("Start Loading Data...")

    mondayWorkingHours = 'Monday-WorkingHours.pcap_ISCX.csv'
    tuesdayWorkingHours = 'Tuesday-WorkingHours.pcap_ISCX.csv'
    wednesdayWorkingHours = 'Wednesday-workingHours.pcap_ISCX.csv'
    thursdayMorningHours = 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv'
    thursdayAfternoonHours = 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'
    fridayMorningFile = 'Friday-WorkingHours-Morning.pcap_ISCX.csv'
    fridayAfternoonPortScan = 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'
    fridayAfternoonDDoS = 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'

    bot = readData(dataroot, fridayMorningFile, 'Bot')
    benign = readData(dataroot, mondayWorkingHours, 'Benign')
    ssh_ftp = readData(dataroot, tuesdayWorkingHours, 'SSH_FTP')
    dos_hulk = readData(dataroot, wednesdayWorkingHours, 'DoS Hulk')
    web = readData(dataroot, thursdayMorningHours, 'Web Attack')
    infiltration = readData(dataroot, thursdayAfternoonHours, 'Infiltration')
    ddos = readData(dataroot, fridayAfternoonDDoS, 'DDoS')
    port_scan = readData(dataroot, fridayAfternoonPortScan, 'PortScan')

    data = pd.concat([benign, bot, ddos, dos_hulk, infiltration, port_scan, ssh_ftp, web], ignore_index=True)
    train_X = data.drop(columns=['Label'])
    train_y = data[['Label']]
    #print(train_X.head())
    #print(train_y['Label'].value_counts())
    #labelencoder = LabelEncoder()
    #X = data.iloc[:, :-1]
    #y = data.iloc[:,-1]
    
    
    #y = labelencoder.fit_transform(y)
    #y = data.iloc[:,-1]
    #y = labelencoder.fit_transform(y)
    X_train, y_train = balanceData(train_X,train_y)
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
