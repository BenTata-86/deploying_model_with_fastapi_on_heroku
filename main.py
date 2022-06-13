import os
import joblib
from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from src.ml import preprocess_data
from src.ml import model

app = FastAPI()

if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

#class Person(BaseModel):


@app.get("/")
async def root():
    return {"message": "Welcome to the App"}

#@app.post("/income_class")
#async def income_prediction():

