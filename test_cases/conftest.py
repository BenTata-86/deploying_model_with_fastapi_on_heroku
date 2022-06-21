import pytest
import os
import pandas as pd
from fastapi.testclient import TestClient
from src.train_model import train_save_model
from src.ml.preprocess_data import process_data
from main import app


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

@pytest.fixture
def data():

    path = 'data/cooked_data.csv.dvc'
    df = pd.read_csv(path)

    return df

@pytest.fixture
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

@pytest.fixture
def processing_data(data, cat_features):
    """
        returns (X, y, encoder, lb)
    """
    (X, y, encoder, lb) =  process_data(
        data,
        categorical_features=cat_features, 
        label="salary", 
        training=True,
        )
    return (X, y, encoder, lb)

@pytest.fixture
def saving_model(processing_data):
    X,y,encoder,_ = processing_data
    pth= f"{ROOT_DIR}/model"

    return train_save_model(X, y, encoder, pth)



@pytest.fixture
def client():
    """
    Get dataset
    """
    api_client = TestClient(app)
    return api_client
