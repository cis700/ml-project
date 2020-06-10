# Name: Fanion Newsome, Ademola Adejokun

# import warnings filter
import os
import sys
import traceback
from os.path import join
from time import time
from random import seed
from random import randint
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer

from warnings import simplefilter
# ignore all future warnings
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV

from src.framework.features.feature_importance import FeatureImportance
from src.framework.features.feature_selection import FeatureSelection
from src.framework.logging.logger import Logger
from src.framework.modeling.models import Models
from src.framework.modeling.stack_by_features import StackByFeatures
from src.framework.utils.utilities import Utilities as utils
from src.modeling.modeling import Modeling

simplefilter(action='ignore', category=FutureWarning)

all_features = [' Destination Port', ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets',
                ' Total Length of Fwd Packets', ' Total Length of Bwd Packets', ' Fwd Packet Length Max',
                ' Fwd Packet Length Min', ' Fwd Packet Length Mean', ' Fwd Packet Length Std', ' Bwd Packet Length Max',
                ' Bwd Packet Length Min', ' Bwd Packet Length Mean', ' Bwd Packet Length Std', ' Flow Bytes/s',
                ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
                ' Fwd IAT Total',
                ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', ' Fwd IAT Min', ' Bwd IAT Total', ' Bwd IAT Mean',
                ' Bwd IAT Std', ' Bwd IAT Max', ' Bwd IAT Min', ' Fwd PSH Flags', ' Bwd PSH Flags', ' Fwd URG Flags',
                ' Bwd URG Flags', ' Fwd Header Length', ' Bwd Header Length', ' Fwd Packets/s', ' Bwd Packets/s',
                ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean', ' Packet Length Std',
                ' Packet Length Variance', ' FIN Flag Count', ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count',
                ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count', ' Down/Up Ratio',
                ' Average Packet Size', ' Avg Fwd Segment Size', ' Avg Bwd Segment Size',
                ' Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk',
                ' Bwd Avg Packets/Bulk', ' Bwd Avg Bulk Rate', ' Subflow Fwd Packets', ' Subflow Fwd Bytes',
                ' Subflow Bwd Packets', ' Subflow Bwd Bytes', ' Init_Win_bytes_forward', ' Init_Win_bytes_backward',
                ' act_data_pkt_fwd', ' min_seg_size_forward', ' Active Mean', ' Active Std', ' Active Max',
                ' Active Min', ' Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min', ' Label']

__release__ = True
dataset_name = 'CICIDS2017'
data_root = 'data/processed/'
dataset_master = '{}_MasterData.csv'.format(dataset_name)
dataset_clean = '{}_CleanedData.csv'.format(dataset_name)
application_log = '{}_logs.txt'.format(dataset_name)

# display complete contents of a dataframe
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
# instantiate objects
sys.stdout = Logger()

# Note: An improvement for the imbalance with the dataset would be to use the balance_data() to balance the record count
# between the benign and malicious records instead of the current functionality
# within the data_trimmer.py file under the 'src/preparation' folder.

# define under sampling dictionary
# used to establish the number of records
# for each class
DownSample = {
    'BENIGN': 25000,  # we only want to down sample this to reduce bias
    'DoS Hulk': 23012,
    'PortScan': 15880,
    'DDoS': 12802,
    'DoS GoldenEye': 10293,
    'FTP-Patator': 7938,
    'SSH-Patator': 5897,
    'DoS slowloris': 5796,
    'DoS Slowhttptest': 5499,
    'Bot': 1966,
    'Web Attack � Brute Force': 1507,
    'Web Attack � XSS': 652,
    'Infiltration': 36,
    'Web Attack � Sql Injection': 21,
    'Heartbleed': 11
}

UpSample = {
    'BENIGN': 25000,  # we only want to down sample this to reduce bias
    'DoS Hulk': 23012,
    'PortScan': 15880,
    'DDoS': 12802,
    'DoS GoldenEye': 10293,
    'FTP-Patator': 7938,
    'SSH-Patator': 5897,
    'DoS slowloris': 5796,
    'DoS Slowhttptest': 5499,
    'Bot': 5000,
    'Web Attack � Brute Force': 5000,
    'Web Attack � XSS': 5000,
    'Infiltration': 5000,
    'Web Attack � Sql Injection': 5000,
    'Heartbleed': 5000
}


class PreProcessing:

    def __int__(self):
        pass

    def remove_columns_by_variance(self, df):
        # define the transform
        transform = VarianceThreshold()

        # transform the input data
        df_output = transform.fit_transform(df)

        return pd.DataFrame(data=df_output)

    def read_data(self):
        t0 = time()

        # read the master file since the clean dataset file does not exist
        master_filepath = join(data_root, dataset_master)
        print("First Time Reading Master Dataset '{}'".format(master_filepath))
        data = pd.read_csv(master_filepath, names=all_features, error_bad_lines=False, low_memory=False, skiprows=1)
        data = data.rename(columns=lambda x: x.strip())

        # convert inf to nan
        pd.set_option('mode.use_inf_as_na', True)

        tt = time() - t0

        print("read_data() took {} seconds".format(utils.convert(round(tt, 3))))
        return data

    # Name: summarize_data
    #
    # Description: Summarizes a provided dataset by displaying detailed information such as its data types,
    # attribute correlation, skew, etc
    #
    def summarize_data(self, data, target_name='Label', target_exists=False):

        print("\nHead of Data: {}".format(data.head(5)))
        print("\nData Shape: {}".format(data.shape))
        print("\nData Types: {}".format(data.dtypes))
        print("\nDescriptive Stats: {}".format(data.describe()))

        # if the provided dataset has a target column, we can display the distribution of
        if target_exists:
            print("Label Distribution of Master Data: \n{}".format(data[target_name].value_counts()))

        # display correlations between attributes
        # A correlation of -1 or 1 shows a full negative / positive correlation respectively
        # A value of 0 shows no correlation at all.
        #
        # Perform Pearson's Correlation Coefficient
        #
        pd.set_option('display.width', 100)
        pd.set_option('precision', 3)
        correlations = data.corr(method='pearson')
        print("\nPearson's Correlation Coefficient: {}", correlations)

        # Skew of Univariate Distributions
        # The skew result will show a negative (left) or positive (right) skew.
        # Values closer to zero show less skew.
        #
        print("\nSkew of Univariate Distributions: {}".format(data.skew()))

    def clean_data(self, df):

        print("Performing Data Cleanup for '{}' dataset".format(dataset_name))

        try:

            t0 = time()
            df = df.dropna()
            df = df.replace(',,', np.nan, inplace=False)
            df.replace('Infinity', 0.0, inplace=True)
            df.replace('NaN', 0.0, inplace=True)

            df['Flow Bytes/s'].replace("Infinity", 0, inplace=True)
            df["Flow Packets/s"].replace("Infinity", 0, inplace=True)
            df["Flow Packets/s"].replace(np.nan, 0, inplace=True)
            df['Flow Bytes/s'].replace(np.nan, 0, inplace=True)

            df["Bwd Avg Bulk Rate"].replace("Infinity", 0, inplace=True)
            df["Bwd Avg Bulk Rate"].replace(",,", 0, inplace=True)
            df["Bwd Avg Bulk Rate"].replace(np.nan, 0, inplace=True)

            df["Bwd Avg Packets/Bulk"].replace("Infinity", 0, inplace=True)
            df["Bwd Avg Packets/Bulk"].replace(",,", 0, inplace=True)
            df["Bwd Avg Packets/Bulk"].replace(np.nan, 0, inplace=True)

            df["Bwd Avg Bytes/Bulk"].replace("Infinity", 0, inplace=True)
            df["Bwd Avg Bytes/Bulk"].replace(",,", 0, inplace=True)
            df["Bwd Avg Bytes/Bulk"].replace(np.nan, 0, inplace=True)

            df["Fwd Avg Bulk Rate"].replace("Infinity", 0, inplace=True)
            df["Fwd Avg Bulk Rate"].replace(",,", 0, inplace=True)
            df["Fwd Avg Bulk Rate"].replace(np.nan, 0, inplace=True)

            df["Fwd Avg Packets/Bulk"].replace("Infinity", 0, inplace=True)
            df["Fwd Avg Packets/Bulk"].replace(",,", 0, inplace=True)
            df["Fwd Avg Packets/Bulk"].replace(np.nan, 0, inplace=True)

            df["Fwd Avg Bytes/Bulk"].replace("Infinity", 0, inplace=True)
            df["Fwd Avg Bytes/Bulk"].replace(",,", 0, inplace=True)
            df["Fwd Avg Bytes/Bulk"].replace(np.nan, 0, inplace=True)

            df["CWE Flag Count"].replace("Infinity", 0, inplace=True)
            df["CWE Flag Count"].replace(",,", 0, inplace=True)
            df["CWE Flag Count"].replace(np.nan, 0, inplace=True)

            df["Bwd URG Flags"].replace("Infinity", 0, inplace=True)
            df["Bwd URG Flags"].replace(",,", 0, inplace=True)
            df["Bwd URG Flags"].replace(np.nan, 0, inplace=True)

            df["Bwd PSH Flags"].replace("Infinity", 0, inplace=True)
            df["Bwd PSH Flags"].replace(",,", 0, inplace=True)
            df["Bwd PSH Flags"].replace(np.nan, 0, inplace=True)

            df["Fwd URG Flags"].replace("Infinity", 0, inplace=True)
            df["Fwd URG Flags"].replace(",,", 0, inplace=True)
            df["Fwd URG Flags"].replace(np.nan, 0, inplace=True)

            # get a count of nulls
            nan_count = df.isnull().sum().sum()
            print('There are {} NaN entries'.format(nan_count))

            if nan_count > 0:
                print("NaN columns: {}".format(nan_count))
                df.fillna(df.mean(), inplace=True)
                print('Completed Filling NaN values..')

            assert df.isnull().sum().sum() == 0, "There should not be any NaN values"

            # check the data types
            if not __debug__:
                print(df.dtypes)
                print(df.shape)

            # summarize the number of unique values in each column
            unique_counts = df.nunique()

            # record columns to delete
            to_del = [i for i, v in enumerate(unique_counts) if v == 1]

            print("Found the following {} column numbers without unique values".format(to_del))
            if len(to_del) > 0:
                # drop useless columns
                print("Dropping non-unique columns: {}".format(df.columns[to_del]))
                df.drop(df.columns[to_del], axis=1, inplace=True)

            print("Dataset Shape After Cleanup: \n{}".format(df.shape))

            tt = time() - t0
            print("Data Cleanup took {} seconds".format(round(tt, 3)))
            print("\nCreating Clean File for '{}'".format(dataset_name))

            y = df['Label'].apply(lambda x: 0 if 'BENIGN' in x else 1)
            df = df.drop(columns=['Label'])
            df = df.astype(dtype='float64')

            return df, y
        except Exception as e:
            track = traceback.format_exc()
            print(str(e))
            print(track)

    def quantile_scale_data(self, dataframe):
        t0 = time()
        print("Running Quantile Scaler of {} Dataset...".format(dataset_name))

        cols_to_scale = dataframe.columns.difference(['Label'])
        q_scaler = QuantileTransformer()

        ds = q_scaler.fit_transform(dataframe[cols_to_scale])
        dataframe[cols_to_scale] = ds
        dataframe = pd.DataFrame(dataframe, columns=dataframe.columns)

        # summarize transformed data
        np.set_printoptions(precision=3)
        print("Quantile Scaler Summary: {}\n".format(ds[0:5, :]))
        print("Finished Running Quantile Scaler...")
        return dataframe

    # Name: standard_scale_data
    # Description: Scales numerical attributes in the provided dataset via Standard scaler.
    #
    def standard_scale_data(self, dataframe):
        print("Running Standard Scaler of {} Dataset...".format(dataset_name))

        cols_to_scale = dataframe.columns.difference(['Label'])
        scaler = StandardScaler()

        ds = scaler.fit_transform(dataframe[cols_to_scale])
        dataframe[cols_to_scale] = ds
        dataframe = pd.DataFrame(dataframe, columns=dataframe.columns)

        # summarize transformed data
        np.set_printoptions(precision=3)
        print("Standard Scaler Summary: {}\n".format(ds[0:5, :]))
        print("Finished Running Standard Scaler...")
        return dataframe

    # Name: min_max_scale_data
    # Description: Scales numerical attributes in the provided dataset via MinMax scaler.
    #
    def min_max_scale_data(self, dataframe):

        print("Running MinMax Scaler on {} Dataset".format(dataset_name))

        # Initialize the MinMax Scalar
        s = MinMaxScaler()

        cols_to_scale = dataframe.columns.difference(['Label'])
        scaler = MinMaxScaler()
        ds = scaler.fit_transform(dataframe[cols_to_scale])
        dataframe[cols_to_scale] = ds
        dataframe = pd.DataFrame(dataframe, columns=dataframe.columns)

        # summarize transformed data
        np.set_printoptions(precision=3)
        print("MinMax Summary: {}\n".format(ds[0:5, :]))
        print("Finished Running MinMax Scaler...")

        return dataframe

    # Name: balance_data
    #
    # Description: Given a training (i.e. attacks) dataset and the target class (i.e. y)
    # we can utilize the RandomUnderSampler and RandomOverSampler to bring balance to the dataset.
    #
    def balance_data(self, attacks, y):

        down_sample = RandomUnderSampler(sampling_strategy=DownSample, random_state=0)
        up_sample = RandomOverSampler(sampling_strategy=UpSample, random_state=0)

        down_sample_attacks, down_sample_attacks_label = down_sample.fit_sample(attacks, y)
        print(pd.DataFrame(data=attacks).shape)
        print(pd.DataFrame(data=down_sample_attacks_label).shape)

        up_sample_attacks, up_sample_attacks_label = up_sample.fit_sample(down_sample_attacks,
                                                                          down_sample_attacks_label)

        up_sample_attacks = pd.DataFrame(data=up_sample_attacks)
        up_sample_attacks_label = pd.DataFrame(data=up_sample_attacks_label)

        print("Up Sample Attacks Shape:\n", up_sample_attacks.shape)
        print("Up Sample Attacks Label Shape: \n", up_sample_attacks_label.shape)

        return up_sample_attacks, up_sample_attacks_label

    def feature_importance_analysis(self, x_train, y_train):

        fi = FeatureImportance()

        # split into 60:40 ratio
        x_train, y_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.3, random_state=0)
        fi.decision_tree_classifier(x_train, y_train, False)
        fi.decision_tree_regression_classifier(x_train, y_train, False)

        # TODO: Fix issue w/ this classifier, index out of bounds error w/ matplotlib
        #fi.extra_trees_classifier(x_train, y_train, False)

        fi.random_forest_classifier(x_train, y_train)

        # TODO: Classifier runs way too long w/ dataset > 100MB
        # fi.knn_classifier(x_train, y_train, k=3, display=False)
        fi.select_top_k_best(x_train=x_train, y_train=y_train, k_best=20)

    def feature_selection_analysis(self, x_train, y_train):

        fs = FeatureSelection()

        # split into 60:40 ratio
        x_train, y_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.33, random_state=0)
        columns = x_train.columns
        to_drop = fs.highest_correlated_features(x_train=x_train, y_train=y_train, top_n=20)

        # Drop Highest Correlated features
        x_new = x_train.drop(x_train[to_drop], axis=1)

        reduced_columns = x_new.columns
        # Select the ones you want
        print("\nDataset Shape After Applying Best Correlated Features - Feature Selection {}".format(x_new.shape))
        print("\nDataset Columns: {}".format(reduced_columns))

        to_drop = fs.random_forest_classifier(x_new, y_train, columns=columns, random_state=0)
        print("Final Columns to Drop: \n{}".format(to_drop))
        # Drop features after RF Classifier
        x_train_final = x_new.drop(x_new[to_drop], axis=1)

        print('\nDataset Shape After Applying Random Forest Classifier - Feature Selection\n{}'
              .format(x_train_final.shape))

        print("Final Selected Features: {}".format(x_train_final.columns))

        return x_train_final.columns

    def load_data(self):

        t0 = time()
        print("Start Loading Data...")

        x_train = self.read_data()

        print("Attempting to clean found Master Dataset dataset")
        x_train, y_train = self.clean_data(x_train)

        print("Preparing to Perform Scaling on Dataset")
        # x_train = self.standard_scale_data(x_train)
        # x_train = self.min_max_scale_data(x_train)

        # Chose to use Quantile scaling because it faired better
        # in regards to outliers and it had a better performance
        #
        x_train = self.quantile_scale_data(x_train)
        self.feature_importance_analysis(x_train, y_train)

        # feature selection
        final_columns = self.feature_selection_analysis(x_train, y_train)
        x_train = x_train[final_columns]

        # saves a clean copy of the dataset to disk
        if __release__:
            output_df = pd.concat([x_train, y_train], axis=1)
            output_df.to_csv(join(data_root, dataset_clean), encoding='utf-8', index=False)
        print("Finished Scaling on Dataset")

        # KNN runs way too slow w/ the dataset
        # models.knn(x_train, y_train)

        if __release__:
            gb_model = Models(x_train, y_train, test_size=0.3, random_state=0, scaler_type=0)
            gb_model.gradient_boost_classifier()

            lr_model = Models(x_train, y_train, test_size=0.3, random_state=0, scaler_type=0)
            lr_model.logistic_regression_model()

        # Run Stack-By-Features Ensemble
        # Attempted to start the test size at 10% and incrementally increase it by 10%
        # for a few iterations until 50% to see if the accuracy would change.  Also added
        # means to remove a random feature per iteration.
        #
        test_size = 0.3
        for i in range(1, 5):
            rand_int = randint(1, 41)
            print("Index value chosen to remove: {}".format(rand_int))
            x_train = x_train.drop(x_train.columns[rand_int], axis=1)
            print("Current # of Columns: {}".format(len(x_train.columns)))
            sbf = StackByFeatures(x_train, y_train, test_size=test_size)
            test_size += 0.1
            print("Current Test Size: {}%".format(test_size * 100))

        # Summarizes the dataset ( displays when run w/ python -O app.py
        if __release__:
            self.summarize_data(x_train, target_exists=False)

        tt = time() - t0
        print("Pre-processing took {} seconds".format(utils.convert(round(tt, 3))))
