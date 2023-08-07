import pandas as pd

file_path = r'E:\Dev\Deakin\Project_Orion\DataScience\Clean Datasets\Geolife Trajectories 1.3\Data\000\Trajectory\20081023025304.plt'

# Define the column names
def read_plt(file_path):

    columns = ['Latitude', 'Longitude', 'Reserved', 'Altitude', 'NumDays', 'Date', 'Time']


    # Read the PLT file, skipping the first six lines
    data = pd.read_csv(file_path, skiprows=6, header=None, names=columns)
    data['Altitude'] = data['Altitude'] * 0.3048
    # Optionally, combine Date and Time into a single datetime column
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
    data.drop(columns=['Date', 'Time'], inplace=True)

    # Now, 'data' is a DataFrame containing the trajectory data
    print(data.head())
def get_trajectory():
    pass
def get_speed():
    pass    
def get_distance():
    pass
def get_acceleration():
    pass

def get_stops():
    pass
def get_mode():
    pass
def get_frequent_areas():
    pass

def get_anomalies():
    pass

def predict_traffic():
    pass

read_plt(file_path)