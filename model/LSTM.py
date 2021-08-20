
from keras import regularizers
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import Flatten
import tensorflow as tf

def LSTM(n_steps, n_features):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(128, input_shape=(n_steps, n_features), return_sequences=True))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5)) 
    model.add(tf.keras.layers.LSTM(64, return_sequences=True))
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5)) 
    model.add(tf.keras.layers.Dropout(0.3)) 
    model.add(tf.keras.layers.LSTM(28, return_sequences=False))
    model.add(tf.keras.layers.Dropout(0.3)) 
    model.add(tf.keras.layers.Dense(1))
    
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())
    return model
