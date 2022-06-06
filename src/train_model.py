import pandas as pd

from joblib import dump

from .ml.basic_cleaning import basic_cleaning
from .ml.data import process_data
from .ml.model import train_model



# Load data.
X = basic_cleaning("../data/census.csv.dvc")

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
# Process the train data with the process_data function
X_train, y_train, encoder, lb = process_data(
    X, categorical_features=cat_features, label="salary", training=True
)

# Train and save a model.

model = train_model(X_train, y_train)

dump(model, f"{root_path}/model/model.joblib")
dump(encoder, f"{root_path}/model/encoder.joblib")
dump(lb, f"{root_path}/model/lb.joblib")
