n_steps=(10 15)
r_window_size=(15 30)
f_wnd_dim=(5 10)
p_wnd_dim=(15 25)
model="LSTM"
CPD_data_type="raw"

for n in "${n_steps[@]}";
do
  for r in "${r_window_size[@]}"; 
  do
    for f in "${f_wnd_dim[@]}";
    do
      for p in "${p_wnd_dim[@]}";
      do
        python main.py --CPD_data_type $CPD_data_type --model $model --n_steps $n --r_window_size $r --f_wnd_dim $f --p_wnd_dim $p
      done
    done
  done
done
  
  