import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from geopy.distance import geodesic
import numpy as np
import utm

class RealTimeTracking:
    def __init__(self, predictive_model):
        self.predictive_model = predictive_model

    def get_trajectory(self, user_id, gps_data):
        return gps_data[gps_data['user_id'] == user_id]

    def get_direction(self, trajectory):
        directions = []
        for i in range(1, len(trajectory)):
            start_point = utm.from_latlon(*trajectory.iloc[i-1][['Latitude', 'Longitude']])
            end_point = utm.from_latlon(*trajectory.iloc[i][['Latitude', 'Longitude']])
            direction = np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
            directions.append(direction)
        return np.mean(directions)

    def get_speed(self, trajectory):
        speeds = []
        for i in range(1, len(trajectory)):
            distance = geodesic(trajectory.iloc[i-1][['Latitude', 'Longitude']].tolist(),
                                trajectory.iloc[i][['Latitude', 'Longitude']].tolist()).meters
            time_diff = (trajectory.iloc[i]['Datetime'] - trajectory.iloc[i-1]['Datetime']).seconds
            speed = distance / time_diff if time_diff != 0 else 0
            speeds.append(speed)
        return speeds

    def get_distance(self, point1, point2):
        return geodesic(point1[['Latitude', 'Longitude']].tolist(), point2[['Latitude', 'Longitude']].tolist()).meters

    def get_acceleration(self, speeds):
        accelerations = []
        for i in range(1, len(speeds)):
            time_diff = 1
            acceleration = (speeds[i] - speeds[i-1]) / time_diff
            accelerations.append(acceleration)
        return accelerations

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
        elif avg_speed < 15:
            return 'cycling'
        else:
            return 'driving'

    def get_frequent_areas(self, trajectory, eps, min_samples):
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

    def preprocess_data(self, trajectory):
        features = [
            [
                point['Datetime'].timestamp(),
                point['Latitude'],
                point['Longitude'],
                self.get_speed(trajectory)[i],
                self.get_direction(trajectory.iloc[i:i+2])
            ]
            for i, point in enumerate(trajectory.iloc[:-1].to_dict('records'))
        ]
        return np.array(features).reshape(-1, self.predictive_model.seq_length, 5)

    def predict_real_time_coordinates(self, trajectory):
        if self.predictive_model.model is None:
            print("No predictive tracking available: no model attached.")
            return None
        features = self.preprocess_data(trajectory)
        predictions = self.predictive_model.model.predict(features)
        return predictions

    def predict_traffic(self, gps_data, eps=10, min_samples=5):
        clusters = self.get_frequent_areas(gps_data, eps, min_samples)
        mean_predicted_coordinates = {}
        for cluster_id, cluster in clusters.items():
            predicted_latitudes = []
            predicted_longitudes = []
            for trajectory in cluster:
                predicted_coordinates = self.predict_real_time_coordinates([trajectory])
                if predicted_coordinates is None:
                    continue
                predicted_lat = np.mean(predicted_coordinates[:, :, 0])
                predicted_long = np.mean(predicted_coordinates[:, :, 1])
                predicted_latitudes.append(predicted_lat)
                predicted_longitudes.append(predicted_long)
            mean_predicted_coordinates[cluster_id] = (np.mean(predicted_latitudes), np.mean(predicted_longitudes))
        return mean_predicted_coordinates





file_path = r'E:\Dev\Deakin\Project_Orion\DataScience\Clean Datasets\Geolife Trajectories 1.3\Data\000\Trajectory\20081023025304.plt'

# this should only be used during testing --some modifications needed
def read_plt(file_path, user_id):
    columns = ['Latitude', 'Longitude', 'Reserved', 'Altitude', 'NumDays', 'Date', 'Time']
    data = pd.read_csv(file_path, skiprows=6, header=None, names=columns)
    data['Altitude'] = data['Altitude'] * 0.3048
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
    data.drop(columns=['Date', 'Time'], inplace=True)
    data['user_id'] = user_id # Add user_id to the DataFrame

    return data

user_id = '000' # Example user ID
trajectory_data = read_plt(file_path, user_id)
real_time_tracking = RealTimeTracking(None)
user_trajectory = real_time_tracking.get_trajectory(user_id, trajectory_data)

    # Now, 'data' is a DataFrame containing the trajectory data
read_plt(file_path)