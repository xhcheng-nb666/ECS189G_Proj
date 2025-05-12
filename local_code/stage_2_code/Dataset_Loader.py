'''
Concrete IO class for a specific dataset
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD
   
import pandas as pd
from local_code.base_class.dataset import dataset

class Dataset_Loader(dataset):
    def __init__(self, dName, dDescription, data_path_train, data_path_test):
        dataset.__init__(self, dName, dDescription)
        self.data_path_train = data_path_train
        self.data_path_test = data_path_test

    def load(self):
        print('loading data...')
        train_df = pd.read_csv(self.data_path_train)
        test_df = pd.read_csv(self.data_path_test)
        
        X_train = train_df.iloc[:, 1:].values / 255.0  # normalize pixel values
        y_train = train_df.iloc[:, 0].values
        
        X_test = test_df.iloc[:, 1:].values / 255.0
        y_test = test_df.iloc[:, 0].values

        return {
            'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test}
        }
