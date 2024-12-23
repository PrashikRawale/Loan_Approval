import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
import logging

# Configure logging to both console and file
log_file = 'data_preprocessing.log'
logger = logging.getLogger()

# Create a log directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Set up the log file handler
file_handler = logging.FileHandler(os.path.join('logs', log_file))
file_handler.setLevel(logging.INFO)

# Set up the console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a log formatter
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Apply the formatter to both handlers
file_handler.setFormatter(log_formatter)
console_handler.setFormatter(log_formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Log configuration
logger.setLevel(logging.INFO)

def load_data():
    """Load training and testing data."""
    try:
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.info("Data loaded successfully.")
        return train_data, test_data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_data(train_data, test_data):
    """Preprocess the data by encoding categorical variables and scaling numerical features."""
    try:
        # Separate features (X) and target variable (y)
        X_train = train_data.drop(columns=['loan_status'])
        y_train = train_data['loan_status']
        X_test = test_data.drop(columns=['loan_status'])
        y_test = test_data['loan_status']
        logger.info("Data separated into features and target.")

        # One-Hot Encoding of categorical features
        X_train_encoded = pd.get_dummies(X_train, drop_first=False)  # drop_first=False keeps all categories
        X_test_encoded = pd.get_dummies(X_test, drop_first=False)
        logger.info("One-Hot Encoding applied.")

        # Align Test Data with Train Data Columns (to ensure both datasets have the same columns)
        X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
        logger.info("Test data columns aligned with train data.")

        # Fill missing values with 0
        X_train_encoded.fillna(0, inplace=True)
        X_test_encoded.fillna(0, inplace=True)
        logger.info("Missing values filled with 0.")

        # MinMax Scaling for numerical columns
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_encoded)
        X_test_scaled = scaler.transform(X_test_encoded)
        logger.info("MinMax Scaling applied.")

        # Convert the scaled arrays back to DataFrames with the original column names
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_encoded.columns)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test_encoded.columns)
        logger.info("Scaled data converted to DataFrame.")

        return X_train_scaled, X_test_scaled, y_train, y_test

    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
        raise

def save_data(X_train_scaled, X_test_scaled, y_train, y_test):
    """Save the processed data."""
    try:
        # Recombine preprocessed features with target variable
        train_preprocessed = pd.concat([X_train_scaled, y_train], axis=1)
        test_preprocessed = pd.concat([X_test_scaled, y_test], axis=1)

        # Create the processed data folder if it doesn't exist
        data_path = os.path.join('data', 'processed')
        os.makedirs(data_path, exist_ok=True)

        # Save to CSV
        train_preprocessed.to_csv(os.path.join(data_path, 'train_processed.csv'), index=False)
        test_preprocessed.to_csv(os.path.join(data_path, 'test_processed.csv'), index=False)
        logger.info("Processed data saved successfully.")
    
    except Exception as e:
        logger.error(f"Error saving processed data: {e}")
        raise

def main():
    """Main function to load, preprocess, and save the data."""
    try:
        # Step 1: Load the data
        train_data, test_data = load_data()

        # Step 2: Preprocess the data
        X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data(train_data, test_data)

        # Step 3: Save the processed data
        save_data(X_train_scaled, X_test_scaled, y_train, y_test)

    except Exception as e:
        logger.error(f"An error occurred during processing: {e}")
        raise

# Run the main function
if __name__ == "__main__":
    main()
