import pandas as pd
import numpy as np
import os
import yaml
import logging
from sklearn.model_selection import train_test_split

# Configure logging to both console and file
log_file = 'data_processing.log'
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

def load_data(file_path):
    """Loads CSV data into a DataFrame."""
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Data loaded from {file_path}.")
        return df
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path}. Error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading file: {file_path}. Error: {e}")
        raise

def clean_columns(df):
    """Cleans the DataFrame columns by stripping extra spaces."""
    try:
        df.columns = df.columns.str.strip()
        logger.info("Column names cleaned.")
        return df
    except Exception as e:
        logger.error(f"Error cleaning columns. Error: {e}")
        raise

def split_data(df):
    """Splits the data into features (x) and target (y)."""
    try:
        x = df.drop(['loan_id', 'loan_status'], axis=1)
        y = df['loan_status']
        logger.info("Data split into features and target.")
        return x, y
    except KeyError as e:
        logger.error(f"Column missing in data: {e}")
        raise

def get_test_size():
    """Loads test size from YAML config."""
    try:
        with open('C:\\Users\\N51907\\Documents\\Cookie\\loan_approval\\params.yaml', 'r') as file:
            test_size = yaml.safe_load(file)['data_ingestion']['test_size']
        logger.info(f"Test size loaded: {test_size}.")
        return test_size
    except FileNotFoundError as e:
        logger.error(f"Config file not found: params.yaml. Error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading test size from YAML. Error: {e}")
        raise

def perform_train_test_split(x, y, test_size):
    """Performs train-test split."""
    try:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)
        logger.info(f"Data split into train and test sets with test_size={test_size}.")
        return x_train, x_test, y_train, y_test
    except ValueError as e:
        logger.error(f"Error during train-test split. Error: {e}")
        raise

def save_data(train_data, test_data, data_path):
    """Saves the train and test data locally."""
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(data_path, 'test.csv'), index=False)
        logger.info(f"Data saved at {data_path}.")
    except Exception as e:
        logger.error(f"Error saving data. Error: {e}")
        raise

def main():
    """Main function to load, process, and save the loan approval dataset."""
    try:
        df = load_data("C:\\Users\\N51907\\Documents\\Cookie\\loan_approval\\data\\external\\loan_approval_dataset.csv")
        df = clean_columns(df)
        x, y = split_data(df)
        test_size = get_test_size()
        x_train, x_test, y_train, y_test = perform_train_test_split(x, y, test_size)
        train_data = pd.concat([x_train, y_train], axis=1)
        test_data = pd.concat([x_test, y_test], axis=1)
        data_path = os.path.join('data', 'raw')
        save_data(train_data, test_data, data_path)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise

# Run the main function
if __name__ == "__main__":
    main()
