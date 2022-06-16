import pytest
from fastapi.testclient import TestClient
from main import app



@pytest.fixture
def client():
    """
    Get dataset
    """
    api_client = TestClient(app)
    return api_client


def test_get(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"message": "Welcome to the App"}


def test_post_above(client):
    request = client.post("/", json={'age': 40,
                                     'workclass': 'Private',
                                     'fnlgt': 149449,
                                     'education': 'Bachelors',
                                     'marital_status': 'Married-civ-spouse',
                                     'occupation': 'Exec-managerial',
                                     'relationship': 'Husband',
                                     'race': 'White',
                                     'sex': 'Male',
                                     'hoursPerWeek': 45,
                                     'nativeCountry': 'United-States'
                                     })
    assert request.status_code == 200
    assert request.json() == {"prediction": "above 50K$"}


def test_post_below(client):
    request = client.post("/", json={'age': 18,
                                     'workclass': 'Private',
                                     'fnlgt': 14184,
                                     'education': 'HS-grad',
                                     'marital_status': 'Never-married',
                                     'occupation': 'Sales',
                                     'relationship': 'Not-in-family',
                                     'race': 'Black',
                                     'sex': 'Male',
                                     'hoursPerWeek': 40,
                                     'nativeCountry': 'Cuba'
                                     })
    assert request.status_code == 200
    assert request.json() == {"prediction": "equal or below  50K$'"}
