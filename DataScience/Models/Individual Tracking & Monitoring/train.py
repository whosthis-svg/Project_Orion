from sklearn.externals import joblib

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

    def train_model(self):
        # Train the model on available data
        # Save training date
        pass

    def evaluate_model(self, new_model):
        # Compare new_model to current self.model
        # If better, replace self.model
        pass

    def retrain_model(self, x_days):
        # If last_trained_date is older than x_days, call train_model()
        pass

    def predict_traffic(self):
        # Use the model to predict traffic
        pass

    def save_model(self):
        # Save model and last_trained_date to a file
        joblib.dump((self.model, self.last_trained_date), self.model_path)
