import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import yaml
import logging
import pickle
import os

# Configure logging
log_file = 'model_building.log'
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
    """Load training data."""
    try:
        train_data = pd.read_csv('./data/processed/train_processed.csv')
        logger.info("Training data loaded successfully.")
        return train_data
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        raise

def extract_features_and_target(train_data):
    """Extract features and target from the training data."""
    try:
        x_train = train_data.iloc[:, 0:-1].values
        y_train = train_data.iloc[:, -1].values
        logger.info("Features and target variable separated.")
        return x_train, y_train
    except Exception as e:
        logger.error(f"Error separating features and target: {e}")
        raise

def get_model_parameters():
    """Load model parameters from YAML file."""
    try:
        config = yaml.safe_load(open('params.yaml', 'r'))
        random_state = config['model_building']['random_state']
        n_estimators = config['model_building']['n_estimators']
        logger.info(f"Model parameters loaded: random_state={random_state}, n_estimators={n_estimators}")
        return random_state, n_estimators
    except Exception as e:
        logger.error(f"Error loading model parameters: {e}")
        raise

def build_model(x_train, y_train, random_state, n_estimators):
    """Build and train the RandomForest model."""
    try:
        rf = RandomForestClassifier(random_state=random_state, n_estimators=n_estimators)
        rf.fit(x_train, y_train)
        logger.info("Model trained successfully.")
        return rf
    except Exception as e:
        logger.error(f"Error training the model: {e}")
        raise

def save_model(rf):
    """Save the trained model to a file."""
    try:
        pickle.dump(rf, open('models/model.pkl', 'wb'))  # wb -> binary write mode
        logger.info("Model saved successfully to 'model.pkl'.")
    except Exception as e:
        logger.error(f"Error saving the model: {e}")
        raise

def main():
    """Main function to load data, train the model, and save it."""
    try:
        # Load data
        train_data = load_data()

        # Extract features and target
        x_train, y_train = extract_features_and_target(train_data)

        # Load model parameters
        random_state, n_estimators = get_model_parameters()

        # Build and train the model
        rf = build_model(x_train, y_train, random_state, n_estimators)

        # Save the trained model
        save_model(rf)

    except Exception as e:
        logger.error(f"An error occurred during model building: {e}")
        raise

# Run the main function
if __name__ == "__main__":
    main()
