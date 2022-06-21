from main import app
import os
from sklearn.ensemble import RandomForestClassifier
from src.ml.model import inference, compute_model_metrics

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



def test_get(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to the App"}


def test_post_above(client):
    data = {
        'age': 40,
        'workclass': 'Private',
        'fnlgt': 149596,
        'education': 'Bachelors',
        'marital-status': 'Married-civ-spouse',
        'occupation': 'Exec-managerial',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'hours-per-week': 45,
        'native-country': 'United-States'
        }
    request = client.post("/income_class", json=data)
    assert request.status_code == 200
    assert request.json() == {"prediction": "above 50K$"}


def test_post_below(client):
    request = client.post("/income_class", json={'age': 18,
                                                 'workclass': 'Private',
                                                 'fnlgt': 14184,
                                                 'education': 'HS-grad',
                                                 'marital-status': 'Never-married',
                                                 'occupation': 'Sales',
                                                 'relationship': 'Not-in-family',
                                                 'race': 'Black',
                                                 'sex': 'Male',
                                                 'hours-per-week': 40,
                                                 'native-country': 'Cuba'
                                                 })
    assert request.status_code == 200
    assert request.json() == {"prediction": "equal or below  50K$"}
