import os
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split

from ml.preprocess_data import process_data
from ml.model import *



# Load data.
def load_data(path):
    X = pd.read_csv(path)
    return X



# Train and save a model.

def train_save_model(X_train, y_train, encoder, lb, pth):

    model = train_model(X_train, y_train)

    dump(model, f"{pth}/model.pkl")
    dump(encoder,f"{pth}/encoder.pkl")
    dump(lb, f"{pth}/lb.pkl")
    return model


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
    train.to_csv(f"{ROOT_DIR}/data/train.csv")
    test.to_csv(f"{ROOT_DIR}/data/test.csv")
    #Process Data
    X_train, y_train, encoder, lb = process_data(
        X=train, categorical_features=cat_features, label="salary", training=True)
    #Train Data and save model and encoders
    model=train_save_model(
        X_train=X_train, y_train=y_train, encoder=encoder, lb=lb, pth=model_pth
        )
    preds=inference(model=model, X=X_train)
    precision,_,-= compute_model_metrics(y=y_train, preds=preds)
    print(precision)
    

if __name__ == "__main__":
    main()