import os
import numpy as np
import pandas as pd
import logging
from src.utils.data_ingestion import load_data, download_dataset, extract_dataset
from src.utils.data_preprocessing import prepare_data, create_preprocessor, transform_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main(data_dir="data", train_file="data/Dataset/train.csv"):
    try:
        if not os.path.exists(data_dir):
            logger.info(f"Data directory '{data_dir}' not found. Downloading and extracting dataset.")
            download_dataset()
            extract_dataset()

        # Load the training dataset
        logger.info(f"Loading training dataset from '{train_file}'.")
        train_data = load_data(train_file)
        train_data = train_data.dropna()

        # Convert the date to datetime format and set as index
        logger.info("Converting 'Date' column to datetime format and setting it as index.")
        train_data['Date'] = pd.to_datetime(train_data['Date'])
        train_data.set_index('Date', inplace=True)

        # Prepare the data by splitting into training and validation sets
        logger.info("Preparing data by splitting into training and validation sets.")
        X_train, X_val, y_train, y_val = prepare_data(train_data)

        # Create the preprocessor for numerical and categorical features
        logger.info("Creating data preprocessor.")
        preprocessor = create_preprocessor()

        # Transform the training and validation data
        logger.info("Transforming training and validation data.")
        X_train_transformed, X_val_transformed = transform_data(X_train, X_val, preprocessor)

        logger.info("Feature pipeline execution complete.")
        return X_train_transformed, X_val_transformed, y_train, y_val

    except Exception as e:
        logger.error(f"An error occurred during the feature pipeline execution: {e}")
        raise

if __name__ == "__main__":
    main()