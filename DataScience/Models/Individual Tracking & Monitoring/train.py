import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Dropout
from sklearn.externals import joblib
from datetime import datetime

class PredictiveTracking:
    def __init__(self, gps_data, model_path="model.pkl", seq_length=20, pred_length=5):
        self.gps_data = gps_data
        self.model_path = model_path
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.load_model()
        self.load_data()

    def load_model(self):
        try:
            self.model, self.last_trained_date = joblib.load(self.model_path)
        except:
            self.model = None
            self.last_trained_date = None
            
    def create_sequences(self, data):
        X, y = [], []
        for i in range(len(data) - self.seq_length - self.pred_length + 1):
            seq_in = data[i:i + self.seq_length, :]
            seq_out = data[i + self.seq_length:i + self.seq_length + self.pred_length, 1:3] # lat and long columns
            X.append(seq_in)
            y.append(seq_out)
        return np.array(X), np.array(y)

    def load_data(self):
        # Load data from gps_data (in the form of a CSV file [time, lat, long, speed, direction])
        data = pd.read_csv(self.gps_data).values
        X, y = self.create_sequences(data)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        # Train the model on available data (LSTM)
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(self.seq_length, 5))) # 5 features
        self.model.add(Dropout(0.2))
        self.model.add(Bidirectional(LSTM(50, return_sequences=True)))
        self.model.add(Dropout(0.2))
        self.model.add(Bidirectional(LSTM(50, return_sequences=True)))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(2)) # 2 output features (lat and long)
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(self.X_train, self.y_train, epochs=10)
        self.last_trained_date = datetime.now()

    def evaluate_model(self, new_model):
        # Compare new_model to current self.model
        current_loss = self.model.evaluate(self.X_test, self.y_test)
        new_loss = new_model.evaluate(self.X_test, self.y_test)
        if new_loss < current_loss:
            self.model = new_model

    def predict_traffic(self):
        # Use the model to predict traffic (coordinates)
        return self.model.predict(self.X_test)

    def save_model(self):
        # Save model and last_trained_date to a file
        joblib.dump((self.model, self.last_trained_date), self.model_path)
                  