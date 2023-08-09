import os
import glob
from sklearn.preprocessing import MinMaxScaler
import wandb
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint
from tensorflow.keras.layers import LSTM
import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Masking
import keras
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print("Keras version:", keras.__version__)
print("Tensorflow version:", tf.__version__)
print("numpy version:", np.__version__)
print("pandas version:", pd.__version__)
print("scikit-learn version:", sklearn.__version__)

# if tf.__version__ != '2.10.0':
#     raise Exception('Incorrect TensorFlow version')

# # Check CUDA version
# if tf.sysconfig.get_build_info()['cuda_version'] != '11.2':
#     raise Exception('Incorrect CUDA version')

# # Check cuDNN version
# if tf.sysconfig.get_build_info()['cudnn_version'] != '8.1':
#     raise Exception('Incorrect cuDNN version')

wandb.init(
    # set the wandb project where this run will be logged
    project="Project Orion",

    # track hyperparameters and run metadata with wandb.config
    config={
        "layer_1": 100,
        "activation_1": "tanh",
        "optimizer": "adam",
        "loss": "sparse_categorical_crossentropy",
        "metric": "accuracy",
        "epoch": 30,
        "batch_size": 64
    }
)

class TrainingClass:
    def __init__(self, data_path=r'E:\Test\A_DeviceMotion_data'):
        self.data_path = data_path
        self.model = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))
        print("Num TPUs Available: ", len(tf.config.list_physical_devices('TPU')))

    def load_data(self):
        sequences = []
        labels = []
        max_length = 0

        # Determine max_length without loading all data
        for folder_path in glob.glob(os.path.join(self.data_path, '*_*')):
            for subject_file in glob.glob(os.path.join(folder_path, 'sub_*.csv')):
                with open(subject_file, 'r') as file:
                    row_count = sum(1 for row in file)-1
                    max_length = max(max_length, row_count)

        # Loop through all files in the data path and load with padding
        for folder_path in glob.glob(os.path.join(self.data_path, '*_*')):
            action = os.path.basename(folder_path).split('_')[0]
            for subject_file in glob.glob(os.path.join(folder_path, 'sub_*.csv')):
                df = pd.read_csv(subject_file)
                df = df.iloc[:, 1:]
                # Pad the DataFrame with zeros up to max_length rows
                padded_df = pd.DataFrame(index=range(max_length), columns=df.columns).fillna(0)
                padded_df.iloc[:len(df)] = df.values
                sequences.append(padded_df.values)
                labels.append(action)

        print("Loaded {} sequences".format(len(sequences)))
        X = np.stack(sequences)
        y = LabelEncoder().fit_transform(labels)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.X_train = self.X_train.astype('float')
        self.X_test = self.X_test.astype('float')
        print(self.X_train.shape, self.X_train.dtype)
        print(self.y_train.shape, self.y_train.dtype)
        print(self.X_test.shape, self.X_test.dtype)
        print(self.y_test.shape, self.y_test.dtype)
        
        
    def preprocess_data(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.X_train = np.array([self.scaler.fit_transform(x) for x in self.X_train])
        self.X_test = np.array([self.scaler.transform(x) for x in self.X_test])

    def train_model(self):
        self.model = Sequential()
        self.model.add(Masking(mask_value=0., input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(LSTM(50, return_sequences=True))  # Return sequences if you want to add more LSTM layers after     
        self.model.add(LSTM(50, return_sequences=False)) # Last LSTM layer should have return_sequences=False if followed by dense layers

        
        self.model.add(Dense(6, activation='tanh'))

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=10)

        self.model.fit(self.X_train, self.y_train, epochs=30, batch_size=64, validation_data=(self.X_test, self.y_test), callbacks=[early_stopping,WandbMetricsLogger(log_freq=5),WandbModelCheckpoint("models")])

    def save_model(self, file_path='model.h5'):
        self.model.save(file_path)

TC = TrainingClass()
TC.load_data()
TC.preprocess_data()
TC.train_model()
TC.save_model()
