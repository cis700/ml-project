import os
from os.path import join
import pandas as pd
import random
import csv
import glob
import numpy as np
from time import time
from src.framework.utils.utilities import Utilities


# The reason for this script is to read the records from the 8 CSV files which store large amounts of 'BENIGN'
# records which we want to trim down to a manageable size (i.e. 25 - 30K) from over 2 million. We utilize the Random
# class to pick an arbitrary count of 'BENIGN' network records we want to trim down to per file This will help with
# the under sampling of the 'BENIGN' records and since we will run an oversampling on the non 'BENIGN' records during
# pre-processing, we do not filter out those records at this stage.

# defined lower bound for generating a random number
ATTACK_LOWER_BOUND = 5200
ATTACK_UPPER_BOUND = 5400

BENIGN_LOWER_BOUND = 5200
BENIGN_UPPER_BOUND = 5400

masterOutputFilename = 'CICIDS2017_MasterData.csv'
processedDataPath = 'data/processed/'
rawDataPath = 'data/raw/'


class DataTrimmer:
    utils = Utilities

    def __init__(self):
        pass

    #
    # Name: getRandomNumber
    #
    # Description: Returns a random number between a lower bound and upper bound
    #
    def get_random_number(self, lower_limit, upper_limit):
        return random.randint(
            lower_limit, upper_limit)

    def read_data(self):
        filenames = [i for i in glob.glob(join(rawDataPath, '*.csv'))]
        print("Combining the following files: \n{}".format(filenames))
        combined_csv = pd.concat([pd.read_csv(file, dtype=object) for file in filenames], sort=False)
        return combined_csv

    #
    # Name: processFile
    #
    # Description: Provided a root folder and filename, we are to create a new file
    # containing a limited amount of 'BENIGN' records in an output file for the preprocessing stage.
    #
    def process_file(self, benign_limit, attack_limit):
        t0 = time()
        write_to_path = join(processedDataPath, masterOutputFilename)

        data = self.read_data()
        rows, cols = data.shape

        print("Combined CSV files contains {} records with {} feature dimension".format(rows, cols))
        data.rename(columns=lambda x: x.strip(), inplace=True)

        # remove duplicate column
        data.drop(data.columns[55], axis=1, inplace=True)

        # based on the upper limit for each attack type (i.e. benign or non-benign)
        # we want to split the data up so that we have a semi-balance for each type
        #
        benign_data = data.loc[data['Label'] == 'BENIGN'].head(benign_limit * 8)
        attack_data = data.loc[data['Label'] != 'BENIGN'].head(attack_limit * 8)
        attack_data.reindex(np.random.permutation(attack_data.index))
        data_frames = [benign_data, attack_data]
        master_data = pd.concat(data_frames)
        del benign_data
        del attack_data
        master_data.to_csv(write_to_path)
        tt = time() - t0

        print("process_file() took {} seconds".format(self.utils.convert(round(tt, 3))))

    #
    # Name: main
    #
    # Description: Main driver for the program to create a master data file which concatenates
    # records from 8 CSV files into a single file.
    #
    def create_master(self):

        print("Preparing to create CICIDS2017 Combined Data File..")
        benign_limit = self.get_random_number(BENIGN_LOWER_BOUND, BENIGN_UPPER_BOUND)
        attack_limit = self.get_random_number(ATTACK_LOWER_BOUND, ATTACK_UPPER_BOUND)

        self.process_file(benign_limit, attack_limit)
        print("Completed ")
