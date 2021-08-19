import os
import datetime
import random

import numpy as np
from numpy import array
from numpy import hstack
from numpy.lib.npyio import load
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
from model.LSTM import LSTM
from data.load_data import load_data
from sklearn.metrics import mean_squared_error as mse
import shap
from data.load_data import split_sequences
from klcpd.klcpd.model import KL_CPD
import findpeaks
from findpeaks import findpeaks

# args
n_steps = 30
n_features = 11
r_window_size = 30

data_path = 'data/' # should contain a train_df.csv, test_df.csv, val_df.csv
all_columns = ["awake","breath_average", "deep", "duration", "hr_average", "hr_lowest",
      "light", "rem", "restless", "temperature_delta","total", "rmssd"] # last col target 
in_columns = ["awake","breath_average", "deep", "duration", "hr_average", "hr_lowest",
      "light", "rem", "restless", "temperature_delta","total"]
if __name__ == '__main__':

    # load data
    X, y, X_val, y_val, X_test, y_test = load_data(data_path, n_steps, all_columns)
    df = pd.read_csv('data/test_df.csv')
    # create model
    model = LSTM(n_steps,n_features)
    model.fit(X, y, epochs=1, batch_size=50, validation_data=(X_val, y_val), shuffle=False)
    # MSE
    y_pred_test = model.predict(X_test)
    print(mse(y_test, y_pred_test))

    # KL-CPD
    KLCPD_columns = ["deep", "hr_average", "rmssd", 'temperature_delta','breath_average','rem', 'daily_shifts']

    for participant in df.participant_id.unique():
        df_sub = df[df['participant_id']==participant]
        df_sub = df_sub.fillna(method='ffill')
        df_sub = df_sub.fillna(method='bfill')

        df_sub = df_sub[KLCPD_columns]
        data = df_sub.values
        standardized_data = MinMaxScaler().fit_transform(data)
        df_sub = pd.DataFrame(standardized_data)
        df_sub.columns = KLCPD_columns
        # Compute running Pearson correlations
        run_corr = np.empty([(df_sub.shape[1]*df_sub.shape[1]-1)/2, df_sub.shape[0]-r_window_size])
        i=0
        for column_1 in df_sub:
            for column_2 in df_sub:
                if (column_1 != column_2) and (column_1+column_2 not in combination):
                    run_corr[i] = df_sub[column_1].rolling(window=r_window_size).corr(df_sub[column_2])[r_window_size:]
                    i+=1
        
        # time series using the running correlations
        ts = run_corr.T
    
        dim, seq_length = ts.shape[1], ts.shape[0]

        # fit KL-CPD model and derive predictions 
        model_kl = KL_CPD(dim)
        model_kl.fit(ts)
        preds = model_kl.predict(ts)


        ts_df = pd.DataFrame(data=ts)

        preds_df =  pd.DataFrame(data=preds)
        # peak detection
        X = preds_df[0]
        # initialize
        fp = findpeaks(lookahead=7)
        results = fp.fit(X)
        # plot
        # fp.plot()
        results = results.get('df')
        results = results.loc[results['peak'] == True]
        a = results['x'].values

        sub_df = sub_df[in_columns]
