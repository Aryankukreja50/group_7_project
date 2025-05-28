# Import necessary libraries
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import scikit-learn modules for preprocessing, splitting, and evaluation
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

# Import TensorFlow and Keras for building the autoencoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Define constants for data and model storage
DATA_PATH = "CICIDS2018.csv"  # Path to the dataset
MODEL_DIR = "models"                  # Directory to save models
os.makedirs(MODEL_DIR, exist_ok=True)  # Create model directory if it doesn't exist

# Set up logging configuration for informative output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# Data preprocessing function
# Loads the CSV, encodes categorical features, scales data, and returns features and labels
def load_and_preprocess(csv_path: str):
    df = pd.read_csv(csv_path)
    # Drop columns not needed for modeling
    df.drop(['Flow ID', 'Timestamp'], axis=1, errors='ignore', inplace=True)
    # Encode labels: 'BENIGN' as 0, others as 1 (attack)
    df['Label'] = df['Label'].map(lambda x: 0 if x == 'BENIGN' else 1)
    # Encode protocol as integer
    df['Protocol'] = LabelEncoder().fit_transform(df['Protocol'])
    # Fill missing values with 0
    df.fillna(0, inplace=True)
    # Separate features and labels
    X = df.drop('Label', axis=1).values
    y = df['Label'].values
    # Scale features to [0, 1] range
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler


# Build the autoencoder model
# input_dim: number of features
# encoding_dims: tuple specifying the size of each encoding layer
# Returns a compiled Keras model
def build_autoencoder(input_dim: int, encoding_dims=(32, 16, 8)):
    inp = Input(shape=(input_dim,), name="input")
    x = inp
    # Encoder part
    for i, dim in enumerate(encoding_dims):
        x = Dense(dim, activation='relu', name=f"enc_{i}")(x)
    # Decoder part (reverse of encoder, except last layer)
    for i, dim in enumerate(reversed(encoding_dims[:-1])):
        x = Dense(dim, activation='relu', name=f"dec_{i}")(x)
    # Output layer for reconstruction
    out = Dense(input_dim, activation='sigmoid', name="reconstruction")(x)
    
    autoenc = Model(inputs=inp, outputs=out, name="Autoencoder")
    autoenc.compile(optimizer='adam', loss='mse')
    return autoenc


# Train the autoencoder model
# Uses early stopping and model checkpointing
# Returns training history and path to best model weights
def train_autoencoder(model, X_train, X_val, model_dir=MODEL_DIR):
    checkpoint_path = os.path.join(model_dir, "autoenc_best.h5")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss')
    ]
    history = model.fit(
        X_train, X_train,  # Autoencoder tries to reconstruct its input
        epochs=100,
        batch_size=256,
        shuffle=True,
        validation_data=(X_val, X_val),
        callbacks=callbacks,
        verbose=2
    )
    return history, checkpoint_path


# Compute reconstruction error for each sample
# Used to detect anomalies (attacks)
def compute_reconstruction_error(model, X):
    recon = model.predict(X)
    return np.mean(np.square(X - recon), axis=1)


# Find anomaly threshold based on reconstruction error percentile
def find_threshold(errors, percentile=95):
    return np.percentile(errors, percentile)


# Evaluate model performance and print metrics/plots
def evaluate(y_true, errors, threshold):
    y_pred = (errors > threshold).astype(int)  # Predict as attack if error > threshold
    logger.info("Accuracy:  %.4f", accuracy_score(y_true, y_pred))
    logger.info("Precision: %.4f", precision_score(y_true, y_pred))
    logger.info("Recall:    %.4f", recall_score(y_true, y_pred))
    logger.info("F1-Score:  %.4f", f1_score(y_true, y_pred))
    logger.info("AUC:       %.4f", roc_auc_score(y_true, errors))
    logger.info("\n" + classification_report(y_true, y_pred, target_names=['BENIGN','ATTACK']))
    
    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# Plot training and validation loss curves
def plot_history(history):
    plt.figure(figsize=(6,4))
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Autoencoder Training Loss")
    plt.legend()
    plt.show()


# MAIN script to run the workflow
# Loads data, trains model, evaluates, and plots results
def main():
    logger.info("Loading and preprocessing data…")
    X, y, scaler = load_and_preprocess(DATA_PATH)
    
    # Separate normal and attack samples
    X_normal = X[y == 0]
    X_rest   = X[y == 1]
    
    # Split normal data into training and validation sets
    X_train, X_val = train_test_split(X_normal, test_size=0.2, random_state=42)
    
    logger.info("Building autoencoder model…")
    autoenc = build_autoencoder(input_dim=X.shape[1])
    autoenc.summary(print_fn=logger.info)
    logger.info("Training autoencoder…")
    history, ckpt = train_autoencoder(autoenc, X_train, X_val)
    logger.info("Plotting training history…")
    plot_history(history)
    autoenc.load_weights(ckpt)
    
    logger.info("Computing reconstruction errors…")
    errors_all = compute_reconstruction_error(autoenc, X)
    threshold = find_threshold(compute_reconstruction_error(autoenc, X_train))
    logger.info("Using threshold (95th percentile): %.6f", threshold)
    
    logger.info("Evaluating on full dataset…")
    evaluate(y, errors_all, threshold)


# Run the main function if this script is executed directly
if __name__ == "__main__":
    main()
