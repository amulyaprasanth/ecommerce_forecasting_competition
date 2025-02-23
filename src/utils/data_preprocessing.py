import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data(data: pd.DataFrame, split_size: float = 0.8):
    """
    Splits the data into training and validation sets.

    Parameters:
    data (pd.DataFrame): The input data containing features and target.
    split_size (float): The proportion of data to be used for training.

    Returns:
    tuple: Split data (X_train, X_val, y_train, y_val)
    """
    logger.info("Preparing data by splitting into training and validation sets.")
    
    # Split the data into features and labels
    X = data.drop("Sales_Quantity", axis=1)
    y = data['Sales_Quantity']

    split_size = int(split_size * len(data))
    X_train, y_train = X[:split_size], y[:split_size]
    X_val, y_val = X[split_size:], y[split_size:]

    logger.info("Data preparation complete.")
    return X_train, X_val, y_train, y_val

def create_preprocessor():
    """
    Creates a preprocessor for numerical and categorical features.

    Returns:
    ColumnTransformer: A preprocessor object for transforming data.
    """
    logger.info("Creating data preprocessor for numerical and categorical features.")
    
    # Define numerical transformer
    numerical_transformer = Pipeline(
        steps=[
            ('scaler', StandardScaler())
        ]
    )

    # Define categorical transformer
    categorical_transformer = Pipeline(
        steps=[
            ('one_hot_encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]
    )

    num_columns = ['Day_of_Week', 'Holiday_Indicator', 'Past_Purchase_Trends', 'Price', 'Discount', 'Competitor_Price']
    cat_columns = ['Category', 'Brand']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num_transformer', numerical_transformer, num_columns),
            ('cat_transformer', categorical_transformer, cat_columns)
        ]
    )

    logger.info("Preprocessor creation complete.")
    return preprocessor

def transform_data(X_train, X_val, preprocessor: ColumnTransformer):
    """
    Transforms the training and validation data using the preprocessor.

    Parameters:
    X_train (pd.DataFrame): The training data.
    X_val (pd.DataFrame): The validation data.
    preprocessor (ColumnTransformer): The preprocessor object.

    Returns:
    tuple: Transformed training and validation data (X_train_transformed, X_val_transformed)
    """
    logger.info("Transforming training and validation data.")
    
    # Transform the data
    X_train_transformed = np.array(preprocessor.fit_transform(X_train))
    X_val_transformed = np.array(preprocessor.transform(X_val))

    logger.info("Data transformation complete.")
    return X_train_transformed, X_val_transformed