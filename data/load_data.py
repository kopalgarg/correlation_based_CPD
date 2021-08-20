import os
import datetime
import random

import numpy as np
from numpy import array
from numpy import hstack
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import torch
import torch.nn as nn
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta 
from sklearn.preprocessing import MinMaxScaler
from scipy.io import arff
from sklearn.metrics import precision_score, roc_auc_score
from scipy.special import expit
from scipy.signal import butter, lfilter, freqz
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data(data_path, n_steps, all_columns):
<<<<<<< Updated upstream

    train_df = pd.read_csv(os.path.join(data_path, 'train_df.csv'),low_memory=False)
    test_df = pd.read_csv(os.path.join(data_path, 'test_df.csv'),low_memory=False)
    val_df = pd.read_csv(os.path.join(data_path, 'val_df.csv'),low_memory=False)
=======
    import pdb; pdb.set_trace()
    train_df = pd.read_csv(data_path, "train_df.csv")
    test_df = pd.read_csv(data_path,"test_df.csv")
    val_df = pd.read_csv(data_path,"val_df.csv")
>>>>>>> Stashed changes

    train_df = train_df[all_columns]
    test_df = test_df[all_columns]
    val_df = val_df[all_columns]

    # pre-processing 
        # forward fill
    train_df = train_df.fillna(method='ffill')
    test_df = test_df.fillna(method='ffill')
    val_df = val_df.fillna(method='ffill')
        # min-max scaling 
    train_df_n = MinMaxScaler().fit_transform(train_df.values)
    train_df = pd.DataFrame(train_df_n)
    test_df_n = MinMaxScaler().fit_transform(test_df.values)
    test_df = pd.DataFrame(test_df_n)
    val_df_n = MinMaxScaler().fit_transform(val_df.values)
    val_df = pd.DataFrame(val_df_n)

    train_df.columns = all_columns
    test_df.columns = all_columns
    val_df.columns = all_columns


    dataset = train_df.values

    # convert into input/output
    X, y = split_sequences(dataset, n_steps)

    dataset_test = test_df.values
    X_test, y_test = split_sequences(dataset_test, n_steps)

    dataset_val = val_df.values
    X_val, y_val = split_sequences(dataset_val, n_steps)

    return X, y, X_val, y_val, X_test, y_val


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the dataset
		if end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)
