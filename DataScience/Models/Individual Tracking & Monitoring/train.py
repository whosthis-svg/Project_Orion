import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense,Bidirectional,Dropout
from sklearn.externals import joblib
from datetime import datetime

class PredictiveTracking:
    def __init__(self, gps_data, model_path="model.pkl"):
        self.gps_data = gps_data
        self.model_path = model_path
        self.load_model()

    def load_model(self):
        try:
            self.model, self.last_trained_date = joblib.load(self.model_path)
        except:
            self.model = None
            self.last_trained_date = None

    def load_data(self):
        # Load data from gps_data(in the form of a csv file[time, lat, long, speed, direction])
        data = pd.read_csv(self.gps_data)
        X = data[['lat', 'long', 'speed', 'direction']]
        y = data['traffic'] # assuming the traffic data is present
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self):
        # Train the model on available data(LSTM)
        self.model = Sequential()
        
        # First Bidirectional LSTM layer
        self.model.add(Bidirectional(LSTM(50, return_sequences=True), input_shape=(self.X_train.shape[1], 1)))
        self.model.add(Dropout(0.2))
        
        # Second Bidirectional LSTM layer
        self.model.add(Bidirectional(LSTM(50, return_sequences=True)))
        self.model.add(Dropout(0.2))
        
        # Third Bidirectional LSTM layer
        self.model.add(Bidirectional(LSTM(50)))
        self.model.add(Dropout(0.2))
        
        # Dense Layer
        self.model.add(Dense(1))
        
        self.model.compile(optimizer='adam', loss='mse')
        self.model.fit(self.X_train.values.reshape(-1, self.X_train.shape[1], 1), self.y_train, epochs=10)
        self.last_trained_date = datetime.now()

    def evaluate_model(self, new_model):
        # Compare new_model to current self.model
        current_loss = self.model.evaluate(self.X_test.values.reshape(-1, self.X_test.shape[1], 1), self.y_test)
        new_loss = new_model.evaluate(self.X_test.values.reshape(-1, self.X_test.shape[1], 1), self.y_test)
        if new_loss < current_loss:
            self.model = new_model

    def predict_traffic(self):
        # Use the model to predict traffic
        return self.model.predict(self.X_test.values.reshape(-1, self.X_test.shape[1], 1))

    def save_model(self):
        # Save model and last_trained_date to a file
        joblib.dump((self.model, self.last_trained_date), self.model_path)
