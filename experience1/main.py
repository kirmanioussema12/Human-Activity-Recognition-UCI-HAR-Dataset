# main.py
import numpy as np
import os
from preprocessing import preprocess_sensor_data
from model import build_cnn_lstm_model
from train import train_with_stratified_kfold

def main():
    print("🚀 Starting Human Activity Recognition Training Pipeline")

    # Setup paths (adjust if your data is elsewhere)
    base_path = r"C:\Users\MSI\Desktop\Mitacs Project\Human Activity Recognition\UCI HAR Dataset\train\Inertial Signals\\"
    labels_path = r"C:\Users\MSI\Desktop\Mitacs Project\Human Activity Recognition\UCI HAR Dataset\train\y_train.txt"

    # Preprocess data
    X_scaled, y_encoded, label_encoder = preprocess_sensor_data(base_path, labels_path)
    input_shape = X_scaled.shape[1:]  # (timesteps, features)
    num_classes = len(np.unique(y_encoded))

    # Step 2: Pass model building function directly (no wrapping)
    model_fn = build_cnn_lstm_model

    # Create folders if they don’t exist
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # Step 3: Train using Stratified K-Fold
    train_with_stratified_kfold(X_scaled, y_encoded, label_encoder, model_fn)

    print("✅ Training completed. Best models and plots are saved.")


if __name__ == "__main__":
    main()
