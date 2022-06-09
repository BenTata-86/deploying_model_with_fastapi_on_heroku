import json
import os
import pandas as pd
import numpy as np
from joblib import dump
from sklearn.model_selection import train_test_split

from ml.preprocess_data import process_data
from ml.model import *



# Load data.
def load_data(path):
    X = pd.read_csv(path)
    return X


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
    model = train_model(X_train=X_train, y_train=y_train)
    dump(model, f"{model_pth}/model.joblib")
    dump(encoder,f"{model_pth}/encoder.joblib")


    #Evaluate Model Metrics on Test data
    X_test, y_test, _,_ = process_data(
    test, cat_features, label='salary',training=False, encoder=encoder,lb=lb)

    test_preds = inference(model=model, X=X_test)

    test_prc, test_rcl, test_beta = compute_model_metrics(
        y=y_test, preds=test_preds
    )
    testing_metric = "Precision: %s " \
                              "Recall: %s FBeta: %s" % (test_prc, test_rcl, test_beta)
    with open(f'{model_pth}/output.txt', 'w') as f:
        f.write(testing_metric + '\n')

    #Evaluate Model on slices of Data

    with open(f"{model_pth}/slice.txt", "w") as f:
        model_metrics=compute_score_per_slice(
            model=model, data=test, encoder=encoder, lb=lb, cat_features=cat_features
        )
        f.write(json.dumps(model_metrics))
    

if __name__ == "__main__":
    main()
