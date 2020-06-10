from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


class Utilities:

    def __int__(self):
        pass

    @staticmethod
    def convert(seconds):
        seconds = seconds % (24 * 3600)
        hour = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60

        return "%d:%02d:%02d" % (hour, minutes, seconds)

    @staticmethod
    def type_impute(dataframe, string_type, column_names):
        print("dataframe type: {}".format(type(dataframe)))
        dataframe = dataframe.replace(string_type, np.nan)
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        dataframe = imp_mean.fit_transform(dataframe.values)
        dataframe = pd.DataFrame(dataframe, columns=column_names)
        print(dataframe.columns)
        return dataframe

    @staticmethod
    def remove_non_unique(dataframe):
        # removes columns w/o unique values
        to_remove = []
        for col in dataframe.columns:
            if len(dataframe[col].unique()) == 1:
                to_remove.append(col)
        dataframe = dataframe.drop(labels=to_remove, axis=1)

        return dataframe


    @staticmethod
    def training_test_split(data, size):
        x_train, x_test, y_train, y_test \
            = train_test_split(data['X'], data['Y'], train_size=size, random_state=None, stratify=data['Y'])
        data = {
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test
        }
        return data

    @staticmethod
    def label_splitter(df):
        # Look for the label, split the label from the frame
        label_loc = df.columns.get_loc('Label')
        array = df.values
        y = array[:, label_loc]
        x = np.delete(array, label_loc, 1)
        # Store the data
        data = {'X': x, 'Y': y}
        #print('Shape of Label Column: ', y.shape, 'Data columns shape: ', x.shape)
        return data

    @staticmethod
    def print_dataframe(df, rows=1, cols=None):
        with pd.option_context('display.max_rows', rows, 'display.max_columns', cols):
            print(df)


