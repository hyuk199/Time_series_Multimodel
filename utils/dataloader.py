import os
import numpy as np
import pandas as pd

def house_Load_data(df, scale = False):
        # min Time to Date and NaN to mean
        df = df.resample('H').mean()
        # df = df.resample('D').mean()
        df = df.fillna(df.mean())

        # we use "dataset_train_actual" for plotting in the end.
        dataset_train_actual = df.copy()
        # create "dataset_train for further processing
        dataset_train = df.copy()

        # Select features (columns) to be involved intro training and predictions
        dataset_train = dataset_train.reset_index()
        cols = list(dataset_train)[1:dataset_train.shape[1]]

        # Extract dates (will be used in visualization)
        datelist_train = list(dataset_train['dt'])
        datelist_train = [date for date in datelist_train]

        # data stamp - informer Param
        # self.data_stamp = time_features(datelist_train, freq='h')
        
        # To Numpy
        features = dataset_train[cols]
        dataset_train = features.values

        # To Scaling
        if scale:
                features = dataset_train
                data_mean = features.mean(axis=0)
                data_std = features.std(axis=0)
                features = (features-data_mean)/data_std
                dataset_train = features

        return dataset_train


def jena_Load_data(df, scale = False):
        # min Time to Date and NaN to mean
        df['Date Time'] = pd.to_datetime(df['Date Time'])
        df = df.resample('H', on='Date Time').mean()
        df = df.fillna(df.mean())

        # we use "dataset_train_actual" for plotting in the end.
        dataset_train_actual = df.copy()
        # create "dataset_train for further processing
        dataset_train = df.copy()

        # Select features (columns) to be involved intro training and predictions
        cols = list(dataset_train)[1:dataset_train.shape[1]]
        
        # To Numpy
        features = dataset_train[cols]
        data_mean = features.mean(axis=0)
        data_std = features.std(axis=0)
        features = (features-data_mean)/data_std
        dataset_train = features.values

        # To Scaling
        if scale:
                features = dataset_train
                data_mean = features.mean(axis=0)
                data_std = features.std(axis=0)
                features = (features-data_mean)/data_std
                dataset_train = features

        return dataset_train


def geo_Load_data(df, scale = False):
        # min Time to Date and NaN to mean
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.fillna(df.mean())

        # we use "dataset_train_actual" for plotting in the end.
        dataset_train_actual = df.copy()
        # create "dataset_train for further processing
        dataset_train = df.copy()

        # Select features (columns) to be involved intro training and predictions
        cols = list(dataset_train)[1:dataset_train.shape[1]]
        
        # To Numpy
        features = dataset_train[cols]
        dataset_train = features.values

        # To Scaling
        if scale:
                features = dataset_train
                data_mean = features.mean(axis=0)
                data_std = features.std(axis=0)
                features = (features-data_mean)/data_std
                dataset_train = features

        return dataset_train