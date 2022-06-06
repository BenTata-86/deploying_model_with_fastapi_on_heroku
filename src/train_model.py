# Script to train machine learning model.

from .ml.basic_cleaning import basic_cleaning
from .ml.data import process_data
from .ml.model import train_model
from sklearn.model_selection import train_test_split
# Add the necessary imports for the starter code.

# Load data.
data = basic_cleaning(path='../data/census.csv.dvc')
# Train_test split.
train, test = train_test_split(data, test_size=0.20)

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
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test,encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=True
)

# Train and save a model.


