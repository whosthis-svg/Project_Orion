import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from geopy.distance import geodesic
import numpy as np
import utm


class RealTimeTracking:
    def __init__(self, predictive_model):
        self.predictive_model = predictive_model
        
    #returns the trajectory of the user        
    def get_trajectory(self,user_id, gps_data):
        # Filter the GPS data for the specific user ID
        return [point for point in gps_data if point['user_id'] == user_id]
    
    
    
    #Generate the features for the trajectory
    
    def get_direction(self, trajectory):
        directions = []
        for i in range(1, len(trajectory)):
            start_point = utm.from_latlon(*trajectory[i-1]['coordinates'])
            end_point = utm.from_latlon(*trajectory[i]['coordinates'])
            direction = np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
            directions.append(direction)
        return np.mean(directions)

    def get_speed(self,trajectory):
        speeds = []
        for i in range(1, len(trajectory)):
            distance = geodesic(trajectory[i-1]['coordinates'], trajectory[i]['coordinates']).meters
            time_diff = (trajectory[i]['timestamp'] - trajectory[i-1]['timestamp']).seconds
            speed = distance / time_diff if time_diff != 0 else 0
            speeds.append(speed)
        return speeds

    def get_distance(self,point1, point2):
        return geodesic(point1['coordinates'], point2['coordinates']).meters

    def get_acceleration(self,speeds):
        accelerations = []
        for i in range(1, len(speeds)):
            time_diff = 1  # Assuming 1-second intervals, you'll need to adapt this
            acceleration = (speeds[i] - speeds[i-1]) / time_diff
            accelerations.append(acceleration)
        return accelerations
    
    #returns the stops in the trajectory
    def get_stops(self,trajectory, time_threshold):
        stops = []
        for i in range(1, len(trajectory)):
            if self.get_distance(trajectory[i-1], trajectory[i]) == 0 and \
            (trajectory[i]['timestamp'] - trajectory[i-1]['timestamp']).seconds >= time_threshold:
                stops.append(trajectory[i-1])
        return stops

    #returns the mode of the user
    def get_mode(self,speeds, accelerations):
        avg_speed = np.mean(speeds)
        avg_acceleration = np.mean(accelerations)
        # Define logic to determine mode based on speed and acceleration
        # Example:
        if avg_speed < 2:
            return 'walking'
        elif avg_speed < 15:
            return 'cycling'
        else:
            return 'driving'
        
    #returns the frequent areas in the trajectory
    def get_frequent_areas(self,trajectory, eps, min_samples):
        coords = [point['coordinates'] for point in trajectory]
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label != -1: # Ignore noise
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(trajectory[i])
        return clusters

    #returns the anomalies in the trajectory
    def get_anomalies(self,trajectory):
        coords = [point['coordinates'] for point in trajectory]
        model = IsolationForest().fit(coords)
        anomalies = [point for i, point in enumerate(trajectory) if model.predict([coords[i]]) == -1]
        return anomalies
    
    #returns the time feature of the user    
    def get_time_feature(self,cluster):
        # Define logic to extract relevant time features
        # Example:      
        return cluster[0]['timestamp'].hour
    
    
    #prepares the data for the predictive model
    def preprocess_data(self, trajectory):
        # Preprocess the trajectory data to match the format expected by the predictive model
        features = [
            [
                point['timestamp'],
                point['coordinates'][0], # Latitude
                point['coordinates'][1], # Longitude
                self.get_speed(trajectory)[i],
                self.get_direction(trajectory[i:i+2])
            ]
            for i, point in enumerate(trajectory[:-1])
        ]
        
        return np.array(features).reshape(-1, self.predictive_model.seq_length, 5)
    
    
    # uses the predictive model to predict the next coordinates of the user
    def predict_real_time_coordinates(self, trajectory):
        features = self.preprocess_data(trajectory)
        
        # Predict the next coordinates using the trained predictive model
        predictions = self.predictive_model.model.predict(features)

        return predictions

     #returns the predicted coordinates of traffic by analysing the clusters  
    def predict_traffic(self, gps_data, eps=10, min_samples=5):
        # Get clusters
        clusters = self.get_frequent_areas(gps_data, eps, min_samples)
        
        mean_predicted_coordinates = {}
        for cluster_id, cluster in clusters.items():
            predicted_latitudes = []
            predicted_longitudes = []

            # Predict real-time coordinates for each individual trajectory in the cluster
            for trajectory in cluster:
                predicted_coordinates = self.predict_real_time_coordinates([trajectory])
                predicted_lat = np.mean(predicted_coordinates[:, :, 0]) # Assuming latitude is the first value
                predicted_long = np.mean(predicted_coordinates[:, :, 1]) # Assuming longitude is the second value

                predicted_latitudes.append(predicted_lat)
                predicted_longitudes.append(predicted_long)

            mean_predicted_coordinates[cluster_id] = (np.mean(predicted_latitudes), np.mean(predicted_longitudes))

        return mean_predicted_coordinates




file_path = r'E:\Dev\Deakin\Project_Orion\DataScience\Clean Datasets\Geolife Trajectories 1.3\Data\000\Trajectory\20081023025304.plt'

# this should only be used during testing --some modifications needed
# def read_plt(file_path):

#     columns = ['Latitude', 'Longitude', 'Reserved', 'Altitude', 'NumDays', 'Date', 'Time']


#     # Read the PLT file, skipping the first six lines
#     data = pd.read_csv(file_path, skiprows=6, header=None, names=columns)
#     data['Altitude'] = data['Altitude'] * 0.3048
#     # Optionally, combine Date and Time into a single datetime column
#     data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
#     data.drop(columns=['Date', 'Time'], inplace=True)

#     # Now, 'data' is a DataFrame containing the trajectory data
# read_plt(file_path)