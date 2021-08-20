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
from klcpd import KL_CPD
import findpeaks
from findpeaks import findpeaks
import argparse

all_columns = ["awake","breath_average", "deep", "duration", "hr_average", "hr_lowest",
      "light", "rem", "restless", "temperature_delta","total", "rmssd"] # last col target 
input_columns = ["awake","breath_average", "deep", "duration", "hr_average", "hr_lowest",
      "light", "rem", "restless", "temperature_delta","total"]

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Correlation-based CPD')
    parser.add_argument('--model', type=str, default='LSTM', help='Choose "LSTM" or "LightGBM"')
    parser.add_argument('--n_steps', type=int, default=15, help = 'number of days of data used in prediction model')
    parser.add_argument('--r_window_size', type=int, default=11, help = 'window size for running correlations ')
    parser.add_argument('--data_path', type=str, default='data/', help = 'should contain a train_df.csv, test_df.csv, val_df.csv')
    args = parser.parse_args()
    
    # load data
    X, y, X_val, y_val, X_test, y_test = load_data(args.data_path, args.n_steps, all_columns)
    df = pd.read_csv(os.path.join(args.data_path, 'test_df.csv'))
    
    n_features = X.shape[2]
    exp = (args.model+'_'+str(args.r_window_size)+'_'+str(args.n_steps)+'_run01')
    
    # create model
    if args.model == 'LSTM':
        model = LSTM(args.n_steps, n_features = X.shape[2])
        model.fit(X, y, epochs=1, batch_size=50, validation_data=(X_val, y_val), shuffle=False)
    else:
        model = LSTM(args.n_steps, n_features = X.shape[2])
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
        combination = []
        run_corr = np.empty([int(df_sub.shape[1]*(df_sub.shape[1]-1)/2), df_sub.shape[0]-args.r_window_size])
        i=0
        for column_1 in df_sub:
            for column_2 in df_sub:
                if (column_1 != column_2) and (column_1+column_2 not in combination):
                    run_corr[i] = df_sub[column_1].rolling(window=args.r_window_size).corr(df_sub[column_2])[args.r_window_size:]
                    i+=1
                combination.append(column_1+column_2)
                combination.append(column_2+column_1)
        
        # time series using the running correlations
        ts = run_corr.T
        
        dim, seq_length = ts.shape[1], ts.shape[0]

        # fit KL-CPD model and derive predictions 
        model_kl = KL_CPD(dim)
        model_kl.fit(np.nan_to_num(ts, nan= 0.0))
        preds = model_kl.predict(ts)


        ts_df = pd.DataFrame(data=ts)

        preds_df =  pd.DataFrame(data=preds)
        # peak detection
        x = preds_df[0]
        # initialize
        fp = findpeaks(lookahead=4)
        results = fp.fit(x)
        # plot
        # fp.plot()
        results = results.get('df')
        results = results.loc[results['valley'] == True]
        a = results['x'].values
        
        if not os.path.exists(os.path.join('results', exp)): os.mkdir(os.path.join('results', exp))

        sub_df = df[df['participant_id']==participant][all_columns]
        
        if args.model == 'LSTM':
          explainer = shap.DeepExplainer(model, X[:1000])
          top_1 = []
          top_2 = []
          for i in range(len(preds_df[0].values[a])):
            if i <= len(preds_df[0].values[a])-2:
                sub = sub_df[a[i]:a[i+1]]
                if sub.shape[0] >= args.n_steps:
                    
                    X_test,y_test = split_sequences(sub.values, args.n_steps)
                    shap_values = explainer.shap_values(X_test, check_additivity=False)
                    # avg SHAP for all observations
                    shap_average_value = np.abs(shap_values[0]).mean(axis=0)
                    x_average_value = pd.DataFrame(data=X_test.mean(axis=0), columns = all_columns[:-1])
                    shap_values_2D = shap_values[0].reshape(-1,n_features)
                    X_test_2D = X_test.reshape(-1,n_features)
                    x_test_2d = pd.DataFrame(data=X_test_2D, columns = all_columns[:-1])
                    shap.summary_plot(shap_values_2D, x_test_2d, show=False)
                    path = participant+'_'+str(i)+'_summary_plot.png'
                    if not os.path.exists(os.path.join('results', exp, participant)): os.mkdir(os.path.join('results', exp, participant))
                    plt.savefig(os.path.join('results', exp,participant, path), format = "png",dpi = 150,bbox_inches = 'tight')
                    vals= np.abs(shap_values_2D).mean(0)
                    feature_importance = pd.DataFrame(list(zip(all_columns[:-1], vals)),columns=['col_name','feature_importance_vals'])
                    feature_importance.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
                    top_1.append(feature_importance['col_name'].iloc[0])
                    top_2.append(feature_importance['col_name'].iloc[1])
