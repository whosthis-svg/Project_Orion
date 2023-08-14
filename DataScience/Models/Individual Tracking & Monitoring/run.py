import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from geopy import distance
from geopy.distance import geodesic
from geopy.distance import great_circle
import numpy as np
import glob
import os
from tensorflow.keras.models import load_model
from train import PredictiveTracking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import mean_squared_error,mean_absolute_error

class RealTimeTracking:
    def __init__(self,user_id):
        self.user_id = user_id
        self.seq_length = 20
        self.train_date=None
        self.model_path = f"IndiMods/model_{user_id}.h5"
        self.predictive_model=None
       

    def get_trajectory(self, gps_data):
        return gps_data[gps_data['user_id'] == self.user_id]

    def get_direction(self, point1, point2):

        return np.arctan2(point2['Longitude'] - point1['Longitude'], point2['Latitude'] - point1['Latitude'])

    def get_distance(self, point1, point2):
        return distance.distance((point1['Latitude'],point1['Longitude']),(point2['Latitude'],point2['Longitude'])).meters

    def get_speed(self, initialpoint,finalpoint,initialtime,finaltime):
       
        return self.get_distance(initialpoint,finalpoint) / (finaltime - initialtime).seconds


    def get_acceleration(self, initialspeed,finalspeed,initialtime,finaltime):       
       

        return (finalspeed - initialspeed) / (finaltime - initialtime).seconds
    
    def get_stops(self, trajectory, time_threshold):
        stops = []
        for i in range(1, len(trajectory)):
            if self.get_distance(trajectory.iloc[i-1], trajectory.iloc[i]) == 0 and \
            (trajectory.iloc[i]['Datetime'] - trajectory.iloc[i-1]['Datetime']).seconds >= time_threshold:
                stops.append(trajectory.iloc[i-1])
        return stops

    def get_mode(self, speeds, accelerations):
        avg_speed = np.mean(speeds)
        avg_acceleration = np.mean(accelerations)
        if avg_speed < 2:
            return 'walking'
        elif avg_speed < 20:
            return 'cycling'
        else:
            return 'driving'
        


    def get_frequent_areas(self, trajectory, eps=0.01, min_samples=2):
        coords = [point[['Latitude', 'Longitude']].tolist() for _, point in trajectory.iterrows()]
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label != -1:
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(trajectory.iloc[i])
        return clusters

    def get_anomalies(self, trajectory):
        coords = [point[['Latitude', 'Longitude']].tolist() for _, point in trajectory.iterrows()]
        model = IsolationForest().fit(coords)
        anomalies = [point for i, point in enumerate(trajectory.iterrows()) if model.predict([coords[i]]) == -1]
        return anomalies

    def get_time_feature(self, cluster):
        return cluster.iloc[0]['Datetime'].hour
    
    def create_test_data(self, trajectory, common_areas=None):
        # Process the data similarly to the preprocess_data method
        X, _ = self.preprocess_data(trajectory, common_areas)

        # Select one sequence
        test_sequence = X[0]

        # Replace the first four features with placeholder values (e.g., zeros)
        test_sequence[:, :4] = 0

        # Reshape to match the expected input shape
        test_sequence = test_sequence.reshape(1, test_sequence.shape[0], test_sequence.shape[1])

        return test_sequence             
                           
    def get_ground_truth(self, trajectory):
        # Preprocess the data to get the original features, including the coordinates
        _, y = self.preprocess_data(trajectory)

        # Select the corresponding ground truth for the first sequence
        ground_truth = y[0]

        return ground_truth                        
        
    def test_prediction(self, trajectory, common_areas=None):
        # Create test data for the given trajectory
        test_sequence = self.create_test_data(trajectory, common_areas)

        # Load the trained model
        try:
            self.model = load_model(self.model_path)
        except:
            print("No model found. Please train the model first.")
            return
        # Make a prediction using the test sequence
        prediction = self.model.predict(test_sequence)

        # Retrieve the ground truth (actual future coordinates) for the test sequence
        # Assuming you have a method to obtain the ground truth
        ground_truth = self.get_ground_truth(trajectory)

        # Compute metrics
        mae = mean_absolute_error(ground_truth, prediction[0])
        mse = mean_squared_error(ground_truth, prediction[0])

        # Print the prediction results
        print("Predicted coordinates:")
        for coords in prediction[0]:
            print(f"Latitude: {coords[0]}, Longitude: {coords[1]}")

        # Print the metrics
        print(f"Mean Absolute Error: {mae}")
        print(f"Mean Squared Error: {mse}")
        
    def preprocess_data(self, trajectory, common_areas=None):
        frequent_areas = list(self.get_frequent_areas(trajectory, 10, 5).values())
        combined_areas = [area[0] for area in frequent_areas[:10]]  # Take the first 10 frequent areas

        # If there are fewer than 10 frequent areas, add common areas
        while len(combined_areas) < 10:
            if common_areas is not None and len(common_areas) > 0:
                combined_areas.append({'Latitude': common_areas[0][0], 'Longitude': common_areas[0][1]})
                common_areas.pop(0)
            else:
                combined_areas.append({'Latitude': 0, 'Longitude': 0})  # Fill with zeros if not enough areas

        features = []
        for i, point in enumerate(trajectory.iloc[:-1].to_dict('records')):
            # Add the coordinates to the feature list along with other features
            feature_vector = [
                point['Latitude'],
                point['Longitude'],
                self.get_speed(point, trajectory.iloc[i+1], point['Datetime'], trajectory.iloc[i+1]['Datetime']),
                self.get_direction(point, trajectory.iloc[i+1]), # Direction
                point['Datetime'].weekday(),
                point['Datetime'].hour,
                point['Datetime'].minute
            ] + [area['Latitude'] for area in combined_areas] + [area['Longitude'] for area in combined_areas]

            features.append(feature_vector)

        # Create sequences and future coordinates
        sequences = []
        future_coordinates = []
        for i in range(len(features) - self.seq_length - 9):  # 9 is to accommodate 10 future coordinates
            sequence = features[i:i+self.seq_length]
            future_coord = [features[j][:2] for j in range(i+self.seq_length, i+self.seq_length+10)]  # Take only Latitude and Longitude
            sequences.append(sequence)
            future_coordinates.append(future_coord)

        # Convert to NumPy arrays
        X = np.array(sequences)
        y = np.array(future_coordinates)

        return X, y



        

    def train_personalised_model(self, trajectory_data, retrain=False):        
        preprocessed_data = self.preprocess_data(trajectory_data)
        self.predictive_model = PredictiveTracking(self.user_id, preprocessed_data,'train')
        if not retrain and self.predictive_model is not None:
            print("Model already exists for this user. Set retrain=True to retrain the model.")
            return
        self.predictive_model.train_model()
        self.predictive_model.save_model()
        
    # def predict_real_time_coordinates(self, trajectory):
        

    #     processed_data = self.preprocess_data(trajectory)
    #     self.predictive_model,self.train_date=PredictiveTracking(self.user_id,processed_data).load_model()
    #     print("Loading model trained on",self.train_date,"...")
    #     print("Predicting real time coordinates...",self.train_date)
    #     if self.predictive_model is None:
    #         print("Model not trained. Use train_personalised_model() to train the model.")
    #         return None
        
    #     predictions = self.predictive_model.predict(processed_data)
    #     return predictions

    def predict_traffic(self, gps_data, eps=10, min_samples=5):
        clusters = self.get_frequent_areas(gps_data, eps, min_samples)
        mean_predicted_coordinates = {}
        for cluster_id, cluster in clusters.items():
            predicted_latitudes = []
            predicted_longitudes = []
            for trajectory in cluster:
                predicted_coordinates = self.predict_real_time_coordinates(trajectory)
                if predicted_coordinates is None:
                    continue
                predicted_lat = np.mean(predicted_coordinates[:, :, 0])
                predicted_long = np.mean(predicted_coordinates[:, :, 1])
                predicted_latitudes.append(predicted_lat)
                predicted_longitudes.append(predicted_long)
            mean_predicted_coordinates[cluster_id] = (np.mean(predicted_latitudes), np.mean(predicted_longitudes))
        return mean_predicted_coordinates






test_file_path = r'E:\Dev\Deakin\Project_Orion\DataScience\Clean Datasets\Geolife Trajectories 1.3\Data\000\Trajectory\20081028003826.plt'


# this should only be used during testing --some modifications needed
def read_plt(file_path, user_id):
    columns = ['Latitude', 'Longitude', 'Reserved', 'Altitude', 'NumDays', 'Date', 'Time']
    data = pd.read_csv(file_path, skiprows=6, header=None, names=columns)
    data['Altitude'] = data['Altitude'] * 0.3048
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
    data.drop(columns=['Date', 'Time'], inplace=True)
    data['user_id'] = user_id # Add user_id to the DataFrame

    return data







# common_areas = [(lat1, lon1), (lat2, lon2), ...] # Example common areas i.e. landmarks,coffee shops









directory_path = r"E:\Dev\Deakin\Project_Orion\DataScience\Clean Datasets\Geolife Trajectories 1.3\Data\000\Trajectory\*.plt"

# Extract the user ID from the directory path (assuming it's the parent folder of "Trajectory")
user_id = os.path.basename(os.path.dirname(os.path.dirname(directory_path)))

# Initialize the RealTimeTracking object with the extracted user ID
real_time_tracking = RealTimeTracking(user_id)

# Loop through the .plt files and process the trajectory data
for file_path in glob.glob(directory_path):
    trajectory_data = read_plt(file_path, user_id)
    user_trajectory = real_time_tracking.get_trajectory(trajectory_data)
    
    # Train the model
real_time_tracking.train_personalised_model(user_trajectory,True)



