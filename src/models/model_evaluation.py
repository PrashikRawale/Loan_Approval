import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, recall_score, precision_score
import json
import logging
import os

# Configure logging
log_file = 'model_evaluation.log'
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

def load_model():
    """Load the trained model from a file."""
    try:
        rf = pickle.load(open('models/model.pkl', 'rb'))
        logger.info("Model loaded successfully.")
        return rf
    except Exception as e:
        logger.error(f"Error loading the model: {e}")
        raise

def load_test_data():
    """Load the test data."""
    try:
        test_data = pd.read_csv('./data/processed/test_processed.csv')
        logger.info("Test data loaded successfully.")
        return test_data
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        raise

def calculate_metrics(y_test, y_pred):
    """Calculate accuracy, precision, and recall."""
    try:
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=' Approved')
        recall = recall_score(y_test, y_pred, pos_label=' Approved')
        
        logger.info(f"Metrics calculated: Accuracy={accuracy}, Precision={precision}, Recall={recall}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        raise

def save_metrics(metrics):
    """Save the metrics to a JSON file."""
    try:
        with open('reports/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=5)
        logger.info("Metrics saved to 'metrics.json'.")
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
        raise

def main():
    """Main function to evaluate the model and save metrics."""
    try:
        # Load the trained model
        rf = load_model()

        # Load the test data
        test_data = load_test_data()
        
        # Separate features and target
        x_test = test_data.iloc[:, 0:-1].values
        y_test = test_data.iloc[:, -1].values

        # Predict on test data
        y_pred = rf.predict(x_test)
        logger.info("Prediction completed on test data.")

        # Calculate evaluation metrics
        metrics = calculate_metrics(y_test, y_pred)

        # Save the metrics to a JSON file
        save_metrics(metrics)

    except Exception as e:
        logger.error(f"An error occurred during model evaluation: {e}")
        raise

# Run the main function
if __name__ == "__main__":
    main()
