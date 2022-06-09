import os
import pytest
import pandas as pd
from requests import session
from ml.model import train_model
from train_model import train_save_model, save_metrics
from ml.preprocess_data import process_data

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture(scope="session")
def data():

    path = 's3://tatacensus/6f/3b87232b567a20d00805ddda4d95eb'
    df = pd.read_csv(path)

    return df


@pytest.fixture(scope=session)
def cat_features():
    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    return categorical_features

@pytest.fixture(scope=session)
def processing_data(data, cat_features):
    """
        returns (X, y, encoder, lb)
    """
    (X, y, encoder, lb) = process_data(
        data,
        categorical_features=cat_features, 
        label="salary", 
        training=True,
        )
    return (X, y, encoder, lb)

@pytest.fixture(scope=session)
def best_model(processing_data):
    X,y,_,_ = processing_data

    return train_save_model(X, y)

@pytest.fixture(scope=session)
def save_model(best_model):

    

