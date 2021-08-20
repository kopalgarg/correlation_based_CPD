
import pandas as pd
import numpy as np
import os

import numpy as np
import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
import scipy.stats
from scipy import signal
from ast import literal_eval
import torch

from sklearn.model_selection import train_test_split

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

pd.set_option('display.max_columns', None)

DATAPATH = "~/datasets/stressrecov"

EXTENSION = "/oura"
daily_stressed = pd.read_parquet("~/stressrecovery/data/processed/survey/daily_stress.parquet")

sleep_df = pd.read_parquet("~/stressrecovery/data/processed/oura/sleep_concat.parquet")

sleep_df['date']=sleep_df['date'].astype(str) 

daily_merged = pd.read_parquet("~/stressrecovery/data/processed/survey/daily_merged.parquet")
daily_merged['date']=daily_merged['date'].astype(str) 

daily_merged["covid_shift_any"] = daily_merged.apply (lambda row: (row.daily_covid_shifts___1 or row.daily_covid_shifts___2 or row.daily_covid_shifts___3 ), axis=1)

weekly = pd.read_csv('~/stressrecovery/data/processed/survey/perceived_stress_scale_pss4.csv')
weekly = weekly[['participant_id', 'pss4_1', 'pss4557_startdate']]
weekly['date'] = pd.to_datetime(weekly['pss4557_startdate'], format='%Y-%m-%d %H:%M:%S.%f')
weekly['date'] = weekly['date'].dt.date


daily_weekly_merged = pd.DataFrame()
for i in weekly.participant_id.unique():
    a = weekly['participant_id']==i
    a = weekly[a]
    b = daily_merged['participant_id']==i
    b = daily_merged[b]
    r = pd.date_range(start=b.date.min(), end=b.date.max())
    a=a.set_index('date').reindex(r).fillna(method='ffill').rename_axis('date').reset_index()
    a=a.set_index('date').reindex(r).fillna(method='bfill').rename_axis('date').reset_index()
    b['date'] = pd.to_datetime(b['date'])
    c= pd.merge(a, b,  how='left', on=["participant_id","date"])
    daily_weekly_merged = pd.concat([daily_weekly_merged,c])
    
    # columns to datetype
daily_weekly_merged['date'] = pd.to_datetime(daily_weekly_merged['date'])
sleep_df['date'] = pd.to_datetime(sleep_df['date'])

df = pd.merge(sleep_df, daily_weekly_merged,  how='left', on=["participant_id","date"])
df['date'] = df["date"].astype(str)
df['date'] = pd.to_datetime(df["date"])


#df = df.dropna(subset = ["daily_shifts", "daily_covid_shifts___1", "daily_covid_shifts___2", "daily_covid_shifts___3"])
df =df.dropna(subset=["hr_5min","rmssd_5min"])
df["shift_any"] = df.apply (lambda row: (row.daily_shifts ==1.0 or row.daily_shifts==2.0), axis=1)
df["covid_shift_any"] = df["covid_shift_any"].astype(bool)

def covidshift(row):
    if row["covid_shift_any"] and row["shift_any"]:
        return True
    elif row["shift_any"] == False:
        return None
    else:
        return False


def array_preprocess_DROP(arr):
    dropped = arr[arr!=0]
    if len(dropped)!=0:
        return arr[arr!=0]
    else:
        return None

df["covidshift"] = df.apply (lambda row: covidshift(row), axis=1)

#Turn to literal arrays and drop 0s (missing values)
df["hr_5min"] = df.apply (lambda row: array_preprocess_DROP(np.array(literal_eval(row.hr_5min))), axis=1)
df["rmssd_5min"] = df.apply (lambda row: array_preprocess_DROP(np.array(literal_eval(row.rmssd_5min))), axis=1)

#DROPPING NANs AGAIN BECAUSE PREVIOUS OPERATION WOULD HAVE GENERATED NANs ie: array of [0,0,0] -> NaN
df =df.dropna(subset=["hr_5min","rmssd_5min"])

df["rmssd_lowest"] = df.apply (lambda row: np.amin(row.rmssd_5min), axis=1)

df["hr_max"] = df.apply (lambda row: np.amax(row.hr_5min), axis=1)
df["rmssd_max"] = df.apply (lambda row: np.amax(row.rmssd_5min), axis=1)

df["hr_1quantile"] = df.apply (lambda row: np.quantile(row.hr_5min, 0.25), axis=1)
df["rmssd_1quantile"] = df.apply (lambda row: np.quantile(row.rmssd_5min, 0.25), axis=1)

df["hr_2quantile"] = df.apply (lambda row: np.quantile(row.hr_5min, 0.50), axis=1)
df["rmssd_2quantile"] = df.apply (lambda row: np.quantile(row.rmssd_5min, 0.50), axis=1)

df["hr_3quantile"] = df.apply (lambda row: np.quantile(row.hr_5min, 0.75), axis=1)
df["rmssd_3quantile"] = df.apply (lambda row: np.quantile(row.rmssd_5min, 0.75), axis=1)

def custom_round(x, base=20):
    return int(base * round(float(x)/base))
df['score_bin']=df['score'].apply(lambda x: custom_round(x, base=20))

df['score_bin'] = df['score_bin'].replace([20],0)
df['score_bin'] = df['score_bin'].replace([40],1)
df['score_bin'] = df['score_bin'].replace([60],2)
df['score_bin'] = df['score_bin'].replace([80],3)
df['score_bin'] = df['score_bin'].replace([100],4)

# use pd.concat to join the new columns with your original dataframe
df = pd.concat([df, pd.get_dummies(df['score_bin'], prefix='score_bin')],axis=1)

# now drop the original 'country' column (you don't need it anymore)
df.drop(['score_bin'],axis=1, inplace=True)

# Load Participant Train/Val/Test Split dictionary
read_dictionary = np.load('participant_splits.npy',allow_pickle='TRUE').item()

train_df = df[df["participant_id"].isin(read_dictionary["train"])]
val_df = df[df["participant_id"].isin(read_dictionary["val"])]
test_df = df[df["participant_id"].isin(read_dictionary["test"])]

train_df.to_csv('data/train_df.csv', index=False)
test_df.to_csv('data/test_df.csv', index=False)
val_df.to_csv('data/val_df.csv', index=False)

