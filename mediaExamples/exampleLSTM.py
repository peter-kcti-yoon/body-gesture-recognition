from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import os
import numpy as np
import time
import tensorflow as tf
from tensorflow.python.client import device_lib 

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# 132+63+63
feature_len = 1660
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,feature_len)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
# model.add(LSTM(128, return_sequences=True, activation='relu'))
# model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])


model.summary()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(device_lib.list_local_devices())
gpus = tf.config.experimental.list_logical_devices('GPU')
print(gpus)

for i in range(2):
    
    x = np.ones((30,feature_len))
    start_ckpt = time.time()
    res = model.predict(np.expand_dims(x,axis=0))
    print(res[0], 'took', time.time()- start_ckpt)

# start_ckpt = time.time()
# res = model.predict(np.expand_dims(x,axis=0))

