#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:59:31 2023

@author: adamelshimi
"""
# https://github.com/WhiskersReneeWe/simpleFastAPIappToHeroku/blob/main/main.py
# https://github.com/drjodyannjones/Deploying-a-Machine-Learning-Model-on-Heroku-with-FastAPI/blob/main/src/app/main.py

from fastapi import FastAPI
# Import Union since our Item object will have tags that can be strings or a list.
from typing import Union
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel, Field
import joblib
import pandas as pd

from ml.data import process_data

app = FastAPI(
    title="Adam's API",
    description="An API used for inference on the Census dataset.",
    version="1.0.0",
)

# Load the model from disk
model = joblib.load("ml/model.joblib")
encoder = joblib.load("ml/encoder.joblib")
lb = joblib.load("ml/lb.joblib")

cat_feat = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
# Declare the data object with its components and their type.
class ConsesusData(BaseModel):
    age: int = Field(...,example=31)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=45781)
    education: str = Field(..., example="Masters")
    education_num: int = Field(..., example=14,alias='education-num')
    marital_status: str = Field(..., example="Married-civ-spouse", alias='marital-status')
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Wife")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Female")
    capital_gain: int = Field(..., example=5000,alias='capital-gain')
    capital_loss: int = Field(..., example=5000,alias='capital-loss')
    hours_per_week: int = Field(..., example=50,alias='hours-per-week')
    native_country: str = Field(..., example="United-States",alias='native-country')



@app.post("/predict")
async def inference(data: ConsesusData):
    payload_dict = data.dict(by_alias=True)
    payload_dataframe = pd.DataFrame(data=payload_dict, index=[0])

    X_test, _, _, _ = process_data(
    payload_dataframe, categorical_features=cat_feat,label=None, training=False,encoder= encoder, lb=lb
    )
    # make prediction
    prediction = model.predict(X_test)[0]

    if prediction:
        sal_pred = '<=50K'
    else:
        sal_pred = '>50K'

    return {'prediction': sal_pred}

@app.get("/")
async def greeting():
    return {'result':'Welcome to Adam Elshimis API.  Enter 14 set attributes of an individual to get a prediction on their salary'}


# @app.get("/predict/{item_id}")
# def get_items(item_id: int, count: int = 1):
#     return {"fetch": f"Fetched {count} of {item_id}"}


