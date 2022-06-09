import json
import os
import pandas as pd
import numpy as np
from joblib import dump
from sklearn import metrics, model_selection
from sklearn.model_selection import train_test_split

from ml.preprocess_data import process_data
from ml.model import *



# Load data.
def load_data(path):
    X = pd.read_csv(path)
    return X

def train_save_model(X, y, encoder, pth):
    model = train_model(X, y)
    dump(model, f"{pth}/model.joblib")
    dump(encoder,f"{pth}/encoder.joblib")

    return model

def save_metrics(metric, pth, file):

    with open(f'{pth}/{file}.txt', 'w') as f:
        f.write(json.dumps(metric) + '\n')



def main():
    #Path for cooked data
    path= 's3://tatacensus/6f/3b87232b567a20d00805ddda4d95eb'

    # Define categorical Features.
    cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    ]
    #Setting root directory
    ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

    #Setting model directory
    model_pth = f'{ROOT_DIR}/model'

    #Load data
    X = load_data(path=path)

    #train and test split data
    train, test = train_test_split(X, test_size=0.20)

    #Process Data
    X_train, y_train, encoder,lb = process_data(
        X=train, categorical_features=cat_features, label="salary", training=True)

    #Train Data and save model and encoders
    model = train_save_model(
        X=X_train, y=y_train, encoder=encoder, pth=model_pth
        )
    
    #Evaluate Model Metrics on Test data
    X_test, y_test, _, _ = process_data(
    test, cat_features, label='salary',training=False, encoder=encoder,lb=lb)

    test_preds = inference(model=model, X=X_test)

    test_prc, test_rcl, test_beta = compute_model_metrics(
        y=y_test, preds=test_preds
    )
    testing_metric = {
        "precision": test_prc, "recall": test_rcl, "fbeta": test_beta
        }
    save_metrics(
        metric=testing_metric, pth=model_pth, file='output'
        )


    #Evaluate Model on slices of Data

    slice_metric= compute_score_per_slice(model=model, \
        data=test, encoder=encoder, lb=lb, cat_features=cat_features
        )
    save_metrics(
        metric=slice_metric, pth=model_pth, file='slice_output'
    )
    

if __name__ == "__main__":
    main()
