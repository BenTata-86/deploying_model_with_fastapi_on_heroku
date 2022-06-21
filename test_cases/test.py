import os
import pytest
import pandas as pd
from requests import session
from sklearn.ensemble import RandomForestClassifier
from src.ml.preprocess_data import process_data
from src.ml.model import *
from src.train_model import train_save_model

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))


# Testing Inference

def test_inference(saving_model, processing_data):
    X, y,_,_ = processing_data
    model = saving_model
    y_pred = inference(model, X)

    assert len(y_pred) == len(y)
    assert y_pred.any() == 1

# Testing model is saved
def test_model(saving_model):
    '''
    Test model and saved model
    '''

    best_model= saving_model
    assert isinstance(best_model, RandomForestClassifier)

    try:
        assert os.path.isfile(f"{ROOT_DIR}/model/model.joblib")
    except AssertionError as err:
        raise err

# Testing model metrics
def test_model_metrics(processing_data, saving_model):
    '''
    Testing model metrics
    '''
    X, y, _, _ = processing_data
    model = saving_model

    preds = inference(model, X)

    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(fbeta, float)