`--model` Choose "LSTM" or "LightGBM" \
`--n_steps` number of days of data used to make predictions 1 day into the future \
`--r_window_size` window size for running correlations  \
`--data_path` should contain a train_df.csv, test_df.csv, val_df.csv \
`--CPD_data_type` :use "raw" or "running_correlations" for CPD \

Run:
`python main.py --CPD_data_type = 'raw' --model 'LightGBM' --n_steps 15 r_window_size 15`


