import os
import shutil
import requests
import logging
import pandas as pd
from zipfile import ZipFile

# Set logging to info level
logging.basicConfig(level=logging.INFO)

# Define the URL for downloading the dataset
url = "https://machinehack-be.s3.amazonaws.com/ecommerce_forecasting_for_sales/Dataset.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIA4OZIV247M2XCWEMS%2F20250223%2Fap-south-1%2Fs3%2Faws4_request&X-Amz-Date=20250223T090335Z&X-Amz-Expires=172800&X-Amz-SignedHeaders=host&X-Amz-Signature=e59140a5b3db6f8bcf806de49479eb22f2f237837c7d929627676c2dc505f73b"

data_dir = "data/"
data_filepath = "data/data.zip"

def download_dataset(url: str = url):
    """
    Download the dataset zip file from the given URL and save it in the 'data' directory.
    Args:
        url: str, URL of the dataset zip file. Defaults to the provided URL.
    Returns:
        None
    """
    if not os.path.exists(data_dir):
        logging.info("Data directory does not exist. Creating it...")
        os.makedirs(data_dir, exist_ok=True)

        response = requests.get(url)
        logging.info("Downloading dataset...")
        with open(data_filepath, "wb") as f:
            f.write(response.content)

        logging.info("Dataset downloaded successfully.")
    else:
        logging.info("Dataset already exists in the data directory.")

def extract_dataset(filepath: str = data_filepath):
    """
    Extract the dataset zip file from the given filepath.
    Args:
        filepath: str, Path of the dataset zip file. Defaults to the provided filepath.
    Returns:
        None
    """
    if os.path.exists(filepath):
        logging.info("Extracting dataset...")
        with ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall("data")
        shutil.rmtree("data/__MACOSX")
        os.remove(filepath)
        logging.info("Dataset extracted successfully.")

def load_data(data_path: str):
    """
    Load the dataset from the given data path.
    
    Args:
        data_path (str): Path to the dataset directory.
    
    Returns:
        pd.DataFrame: Loaded dataset as a pandas DataFrame.
    """
    df = pd.read_csv(data_path)
    logging.info("Dataset loaded successfully.")
    return df