import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_sensor_data(base_path, sensor_name):
    df = pd.read_csv(base_path + sensor_name, delim_whitespace=True, header=None)
    return df.values

def preprocess_sensor_data(base_path, labels_path):
    """
    Loads and preprocesses sensor data for human activity recognition.

    Parameters:
    - base_path (str): Path to the directory containing the 9 inertial signal files.
    - labels_path (str): Path to the file containing the labels (y_train.txt).

    Returns:
    - X_scaled (np.ndarray): Standardized 3D sensor data of shape (samples, timesteps, features).
    - y_encoded (np.ndarray): Encoded labels as integers.
    - label_encoder (LabelEncoder): Trained label encoder (to decode labels if needed).
    """

    # Load all 9 sensor axes
    sensor_files = [
        'body_acc_x_train.txt', 'body_acc_y_train.txt', 'body_acc_z_train.txt',
        'body_gyro_x_train.txt', 'body_gyro_y_train.txt', 'body_gyro_z_train.txt',
        'total_acc_x_train.txt', 'total_acc_y_train.txt', 'total_acc_z_train.txt'
    ]
    
    sensor_data = [load_sensor_data(base_path, name) for name in sensor_files]
    
    # Stack into shape (samples, timesteps, features)
    X = np.stack(sensor_data, axis=2)
    print("X shape:", X.shape)
    
    # Load and encode labels
    y = pd.read_csv(labels_path, delim_whitespace=True, header=None).values.flatten()
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print("Labels:", np.unique(y_encoded))

    # Standardize features
    nsamples, ntimesteps, nfeatures = X.shape
    X_reshaped = X.reshape(-1, nfeatures)  # (samples*timesteps, features)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(nsamples, ntimesteps, nfeatures)

    return X_scaled, y_encoded, label_encoder
