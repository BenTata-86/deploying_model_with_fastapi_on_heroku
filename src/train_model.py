import pandas as pd
from joblib import dump

from ml.preprocess_data import process_data
from ml.model import train_model



# Load data.
def load_data(path):
    X = pd.read_csv(path)
    return X


# Process the train data with the process_data function
def processing_data(X, cat_features):
    
    return process_data(
        X, categorical_features=cat_features, label="salary", training=True)

# Train and save a model.

def train_save_model(X_train, y_train, encoder, pth):

    model = train_model(X_train, y_train)

    dump(model, f"{pth}/model.joblib")
    dump(encoder, f"{pth}/encoder.joblib")


def main():
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
    pth = '/home/bshegitim1/udacity_mlops/deploying_model_with_fastapi_on_heroku/model'

    X = load_data(path=path)

    X_train, y_train, encoder,_ = processing_data(X, cat_features)

    train_save_model(X_train=X_train, y_train=y_train, encoder= encoder, pth=pth)
    

if __name__ == "__main__":
    main()