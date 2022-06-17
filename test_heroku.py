import requests
import json

data = {
            'age': 35,
            'workclass': 'Private',
            'fnlgt': 149184,
            'education': 'Bachelors',
            'marital-status': 'Never-married',
            'occupation': 'Sales',
            'relationship': 'Not-in-family',
            'race': 'White',
            'sex': 'Male',
            'hours-per-week': 60,
            'native-country': 'United-States'
        }

r = requests.post('https://census-income-fastapi-app.herokuapp.com/income_class/', json=data)


print("Response code: %s" % r.status_code)
print("Response body: %s" % r.json())

assert r.status_code == 200
