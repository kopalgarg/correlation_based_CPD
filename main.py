import os
import datetime
import random

import numpy as np
from numpy import array
from numpy import hstack
from numpy.lib.npyio import load
import pandas as pd
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import torch
import torch.nn as nn
import math
from datetime import datetime, timedelta 
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from sklearn.model_selection import train_test_split
from model.LSTM import LSTM
from model.LightGBM import lgbm_train
from data.load_data import load_data
from sklearn.metrics import mean_squared_error as mse
import shap
from data.load_data import split_sequences
#from klcpd import KL_CPD
from klcpd.model import KL_CPD
import findpeaks
from findpeaks import findpeaks
import argparse
import matplotlib.cm as cm
import matplotlib as matplotlib
import scipy
import lightgbm as lgb

all_columns = ["awake","breath_average", "deep", "duration", "hr_average", "hr_lowest",
      "light", "rem", "restless", "temperature_delta","total", "rmssd"] # make sure last col target 
input_columns = ["awake","breath_average", "deep", "duration", "hr_average", "hr_lowest",
      "light", "rem", "restless", "temperature_delta","total"]

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Correlation-based CPD')
    parser.add_argument('--model', type=str, default='LSTM', help='Choose "LSTM" or "LightGBM"')
    parser.add_argument('--n_steps', type=int, default=15, help = 'number of days of data used in prediction model')
    parser.add_argument('--r_window_size', type=int, default=30, help = 'window size for running correlations ')
    parser.add_argument('--data_path', type=str, default='data/', help = 'should contain a train_df.csv, test_df.csv, val_df.csv')
    parser.add_argument('--CPD_data_type', type=str, default='running_correlations', help = 'use "raw" or "running_correlations" for CPD')
    parser.add_argument('--f_wnd_dim', type = int, default= 10, help = 'f window dimension')
    parser.add_argument('--p_wnd_dim', type = int, default= 25, help = 'p window dimension')

    args = parser.parse_args()
    
    # load data
    X, y, X_val, y_val, X_test, y_test = load_data(args.data_path, args.n_steps, all_columns)
    df = pd.read_csv(os.path.join(args.data_path, 'test_df.csv'))
    
    n_features = X.shape[2]
    exp = (args.model+'_'+str(args.n_steps)+'_'+str(args.r_window_size)+'_'+str(args.f_wnd_dim)+'_'+str(args.p_wnd_dim)+'_'+args.CPD_data_type+'_'+all_columns[-1])
    
    # create model
    if args.model == 'LSTM':
        model = LSTM(args.n_steps, n_features = X.shape[2])
        model.fit(X, y, epochs=20, batch_size=50, validation_data=(X_val, y_val), shuffle=False)
        # MSE
        print("Mean Squared Error:")
        y_pred_test = model.predict(X_test)
        print(mse(y_test, y_pred_test))
    else:
        lgbmForecast_df, model, x_train = lgbm_train(df, cols=all_columns,
                trg=all_columns[-1], train_ratio=0.9, valid_ratio=0.09, test_ratio=0.01)
        trg = all_columns[-1]
        # MSE
        print("Mean Squared Error:")
        print(mse(df[trg][-len(lgbmForecast_df):], lgbmForecast_df[trg]))


    # KL-CPD
    KLCPD_columns = ["deep", "hr_average", "rmssd", 'temperature_delta','breath_average','rem']

    for participant in df.participant_id.unique()[:10]: # run this for just first 10 participants, but can change to all
        df_sub = df[df['participant_id']==participant]
        df_sub = df_sub.fillna(method='ffill')
        df_sub = df_sub.fillna(method='bfill')
        df_sub = df_sub[KLCPD_columns]
        data = df_sub.values
        standardized_data = MinMaxScaler().fit_transform(data)
        df_sub = pd.DataFrame(standardized_data)
        df_sub.columns = KLCPD_columns
        # Compute running Pearson correlations
        f, (ax1) = plt.subplots(1, 1 ,sharex='col')
        f.set_figheight(6)
        f.set_figwidth(16)
        combination = []
        run_corr = np.empty([int(df_sub.shape[1]*(df_sub.shape[1]-1)/2), df_sub.shape[0]-args.r_window_size])
        i=0
        for column_1 in df_sub:
            for column_2 in df_sub:
                if (column_1 != column_2) and (column_1+column_2 not in combination):
                    run_corr[i] = df_sub[column_1].rolling(window=args.r_window_size).corr(df_sub[column_2])[args.r_window_size:]
                    i+=1
                    if (abs(scipy.stats.pearsonr(df_sub[column_1], df_sub[column_2])[0]) > 0.25):
                        ax1.plot(df_sub[column_1].rolling(window=args.r_window_size).corr(df_sub[column_2]),
                        label = (column_1,column_2))
                    else:
                        ax1.plot(df_sub[column_1].rolling(window=args.r_window_size).corr(df_sub[column_2]), alpha = .2)
                combination.append(column_1+column_2)
                combination.append(column_2+column_1)
        
        # save running correlations plot
        if not os.path.exists(os.path.join('results', exp)): os.mkdir(os.path.join('results', exp))
        if not os.path.exists(os.path.join('results', exp, participant)): os.mkdir(os.path.join('results', exp, participant))
        ax1.set_ylim([-1, 1])
        ax1.legend();
        ax1.set_title('Pearson r')
        ax1.legend(loc='right');

        f.savefig(os.path.join('results', exp, participant, 'running_correlations.png'), format = "png")
        
        # 7-day rolling average plot
        df_sub[ 'deep_rolling_mean' ] = df_sub.deep.rolling(7).mean()

        df_sub[ 'hr_average_rolling_mean' ] = df_sub.deep.rolling(7).mean()
        df_sub[ 'rmssd_rolling_mean' ] = df_sub.rmssd.rolling(7).mean()

        df_sub[ 'temp_delta_rolling_mean' ] = df_sub.temperature_delta.rolling(7).mean()

        df_sub[ 'breath_average_rolling_mean' ] = df_sub.breath_average.rolling(7).mean()

        df_sub[ 'rem_rolling_mean' ] = df_sub.rem.rolling(7).mean()

        df_sub[args.r_window_size:].plot(subplots = True, legend =True, figsize=(16, 8))
        plt.savefig(os.path.join('results', exp, participant, 'raw_signal.png'), format = "png")
        plt.close(); plt.close()
        
        if args.CPD_data_type == 'raw':
          ts = df_sub[args.r_window_size:].values
        else:
          # time series using the running correlations
          ts = run_corr.T

        dim, seq_length = ts.shape[1], ts.shape[0]
        # fit KL-CPD model and derive predictions 
        model_kl = KL_CPD(dim, p_wnd_dim = args.p_wnd_dim, f_wnd_dim = args.f_wnd_dim)
        model_kl.fit(ts)
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
                    
                    plt.savefig(os.path.join('results', exp,participant, path), format = "png",dpi = 150,bbox_inches = 'tight')
                    plt.close()
                    vals= np.abs(shap_values_2D).mean(0)
                    feature_importance = pd.DataFrame(list(zip(all_columns[:-1], vals)),columns=['col_name','feature_importance_vals'])
                    feature_importance.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
                    top_1.append(feature_importance['col_name'].iloc[0])
                    top_2.append(feature_importance['col_name'].iloc[1])
        
        if args.model == 'LightGBM':
            explainer = shap.TreeExplainer(model = model,feature_perturbation='tree_path_dependent')
            top_1 = []
            top_2 = []
            sub_df = sub_df[all_columns[:-1]]
            for i in range(len(preds_df[0].values[a])):
                if i <= len(preds_df[0].values[a])-2:
                    sub = sub_df[a[i]:a[i+1]]
                    if sub.shape[0] >= args.n_steps:
                        shap_values = explainer.shap_values(X= sub)
                        shap.summary_plot(shap_values=shap_values, features= sub, feature_names=all_columns[:-1], plot_type="violin", show =False)
                        path = participant+'_'+str(i)+'_summary_plot.png'
                        plt.savefig(os.path.join('results', exp,participant, path), format = "png",dpi = 150,bbox_inches = 'tight')
                        plt.close()
                        vals= np.abs(shap_values).mean(0)
                        feature_importance = pd.DataFrame(list(zip(all_columns[:-1],vals)),columns=['col_name','feature_importance_vals'])
                        feature_importance.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
                        top_1.append(feature_importance['col_name'].iloc[0])
                        top_2.append(feature_importance['col_name'].iloc[1])
        # save the feature importance plots 
        features = all_columns[:-1]
        f = range(1,len(features)+1)
        # map values 
        features_dct = dict(zip(features, f))
        features_dct_inv = {v: k for k, v in features_dct.items()}

        top_1 = list(map(features_dct.get, top_1))
        top_2 = list(map(features_dct.get, top_2))

        b = []
        for i in range(len(a)):
            if i <= len(preds_df[0].values[a])-2:
                if a[i+1]-a[i] >= args.n_steps:
                    b.append(a[i])

        def color_map_color(value, cmap_name='Wistia', vmin=0, vmax=1):
            cmap_name = plt.colormaps()[82+value]
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
            cmap = cm.get_cmap(cmap_name)  
            rgb = cmap(norm(abs(value)))[:3] 
            color = matplotlib.colors.rgb2hex(rgb)
            return color

        f, (ax1, ax2, ax3) = plt.subplots(3, 1 ,sharex='col')
        f.set_figheight(8)
        f.set_figwidth(16)
        for i in range(len(top_1)):
          if i == len(top_1)-1:
             ax2.plot(range(b[i], a[-1]), 
                      [0]*len(range(b[i], a[-1])), 
                      c =color_map_color(top_1[i]), linewidth = 10, label = features_dct_inv.get(top_1[i]))
             ax3.plot(range(b[i], a[-1]), 
                      [-0.5]*len(range(b[i], a[-1])), 
                      c =color_map_color(top_2[i]), linewidth = 10, label = features_dct_inv.get(top_2[i]))
          else:
            print(i)
            ax2.plot(range(b[i], b[i+1]), 
                     [0]*len(range(b[i], b[i+1])), 
                     c = color_map_color(top_1[i]), linewidth = 10, label = features_dct_inv.get(top_1[i]))
            ax3.plot(range(b[i], b[i+1]), 
                     [-0.5]*len(range(b[i], b[i+1])), 
                     c = color_map_color(top_2[i]), linewidth = 10, label = features_dct_inv.get(top_2[i]))

        ax2.legend(loc='center right')
        ax3.legend(loc='center right')

        ax1.set_title('MMD')
        ax2.set_title('Most important feature')
        ax3.set_title('2nd most important feature')
        ax1.plot(preds_df)
        ax1.plot(a ,preds_df[0].values[a], "x", color = 'red', markersize = 18)
        f.savefig(os.path.join('results', exp, participant, 'feature_importance.png'), format = "png")
        

 
