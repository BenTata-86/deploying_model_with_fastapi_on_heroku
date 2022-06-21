import pytest
from fastapi.testclient import TestClient
from main import app


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
