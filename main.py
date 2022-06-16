import os
from joblib import load
from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from src.ml.preprocess_data import process_data
from src.ml.model import inference
from sklearn.preprocessing import LabelBinarizer

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__)))

best_model = load('model/model.joblib')
encoder = load('model/encoder.joblib')
lb = LabelBinarizer()

app = FastAPI()


class Person(BaseModel):
    age: int
    workclass : str
    fnlgt : int
    education : str
    marital_status : str = Field(alias="marital-status")
    occupation : str
    relationship : str
    race : str
    sex : str
    HoursPerWeek : int = Field(alias="hours-per-week")
    native_country : str = Field(alias="native-country")

    class Config:
        schema_extra = {
            "example" : {
                "age": 42,
                "workclass": "Private",
                "fnlgt": 150000,
                "education": "Bachelors",
                "marital-status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "hours-per-week": 45,
                "native-country": "United-States"
                }
        }


@app.get("/")
async def root():
    return {"message": "Welcome to the App"}

@app.post("/income_class")
async def income_prediction(info: Person):
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

    input_data = info.dict(by_alias=True)
    df = pd.DataFrame(input_data, index=[0])
    X,_,_,_ = process_data(
        df, cat_features, label=None, training =False,encoder=encoder, lb=None
        )
    predicts = inference(best_model, X)
    if predicts == 0:
        prediction = 'equal or below  50K$'
    else:
        prediction = "above 50K$"

    return {"prediction": prediction}
    

