import sys
import os
from os.path import join

from src.preparation.data_trimmer import DataTrimmer
from src.processing.preprocess import PreProcessing
from src.framework.logging.logger import Logger



def main():
    master_output_filename = 'CICIDS2017_MasterData.csv'
    processed_data_path = 'data/processed/'
    write_to_path = join(processed_data_path, master_output_filename)

    preprocess = PreProcessing()
    data_trimmer = DataTrimmer()

    if not os.path.exists(write_to_path):
        data_trimmer.create_master()

    preprocess.load_data()


if __name__ == "__main__":
    main()
