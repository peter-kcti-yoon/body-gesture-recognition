from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from opt import actions
import os



class Gesturer:
    def __init__(self):
        super().__init__()
        self.log_dir = os.path.join('Logs')
        self.tb_callback = TensorBoard(log_dir=self.log_dir)


    def build(self):
        model = Sequential()
        model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
        # model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,63)))
        model.add(LSTM(128, return_sequences=True, activation='relu'))
        model.add(LSTM(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(len(actions), activation='softmax'))

        # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        # print(model.summary())
        return model

