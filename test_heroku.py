import requests
import json

data = {
            'age': 35,
            'workclass': 'Private',
            'fnlgt': 149184,
            'education': 'HS-grad',
            'marital_status': 'Never-married',
            'occupation': 'Sales',
            'relationship': 'Not-in-family',
            'race': 'White',
            'sex': 'Male',
            'hoursPerWeek': 60,
            'nativeCountry': 'United-States'
    }

r = requests.post('https://census-income-fastapi-app.herokuapp.com/income_class/', data=json.dumps(data))

assert r.status_code == 200

print("Response code: %s" % r.status_code)
print("Response body: %s" % r.json())
