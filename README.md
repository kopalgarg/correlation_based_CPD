Requirements 
```
pip install -r requirements.txt
```

Run
```
python main.py --CPD_data_type = 'raw' --model 'LightGBM' --n_steps 15 r_window_size 15
```

`--model` Choose "LSTM" or "LightGBM" \
`--n_steps` Number of days of data used to make predictions 1 day into the future \
`--r_window_size` Window size for running correlations  \
`--data_path` Path to folder containing train_df.csv, test_df.csv, val_df.csv (use `data/create_dataset.py`) \
`--CPD_data_type` Use "raw" or "running_correlations" for CPD 

Experiment with different KLCPD window sizes and running correlation coefficient window size
```
chmod +x run.sh
./run.sh
```
