import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from geopy.distance import geodesic
import numpy as np
import utm


class RealTimeTracking:
    def get_trajectory(self,user_id, gps_data):
        # Filter the GPS data for the specific user ID
        return [point for point in gps_data if point['user_id'] == user_id]
    
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

    def get_stops(self,trajectory, time_threshold):
        stops = []
        for i in range(1, len(trajectory)):
            if self.get_distance(trajectory[i-1], trajectory[i]) == 0 and \
            (trajectory[i]['timestamp'] - trajectory[i-1]['timestamp']).seconds >= time_threshold:
                stops.append(trajectory[i-1])
        return stops

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

    def get_anomalies(self,trajectory):
        coords = [point['coordinates'] for point in trajectory]
        model = IsolationForest().fit(coords)
        anomalies = [point for i, point in enumerate(trajectory) if model.predict([coords[i]]) == -1]
        return anomalies

    def predict_traffic(self, gps_data):
        features = []
        target = []
        eps = 10
        min_samples = 5

        # Define clusters using DBSCAN or other clustering method
        clusters = self.get_frequent_areas(gps_data, eps, min_samples)

        for cluster in clusters.values():
            avg_speed = np.mean(self.get_speed(cluster))
            
           
            avg_direction = self.get_direction(cluster)
            
            # Extract relevant time features (you might want to define this function based on your requirements)
            time = self.get_time_feature(cluster)
            
            # Combine features
            features.append([len(cluster), avg_speed, avg_direction, time])

            # Define the target variable, e.g., observed traffic volume for the cluster
            traffic_volume = self.get_observed_traffic_volume(cluster)  # Define this function as needed
            target.append(traffic_volume)

        # Create and train the regression model
        model = LinearRegression()
        model.fit(features, target)

        # Here you can predict on new data if required
        # For this example, I'm predicting on the training data itself
        predictions = model.predict(features)

        return predictions





file_path = r'E:\Dev\Deakin\Project_Orion\DataScience\Clean Datasets\Geolife Trajectories 1.3\Data\000\Trajectory\20081023025304.plt'

# Define the column names
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