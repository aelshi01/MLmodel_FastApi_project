# import sys
# sys.path.insert(0, '../')

from fastapi.testclient import TestClient
from main import app
import pandas as pd
import joblib
import json
from json import JSONEncoder
import numpy as np
import requests

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# initiating app
App = TestClient(app)


lessThan50K = {  "age": 28,
  "workclass": "Private",
  "fnlgt": 338409,
  "education": "Bachelors",
  "education-num": 13,
  "marital-status": "Married-civ-spouse",
  "occupation": "Prof-specialty",
  "relationship": "Wife",
  "race": "Black",
  "sex": "Female",
  "capital-gain": 0,
  "capital-loss": 0,
  "hours-per-week": 40,
  "native-country": "Cuba"}

# Load the model from disk
model = joblib.load("ml/model.joblib")
encoder = joblib.load("ml/encoder.joblib")
lb = joblib.load("ml/lb.joblib")



def test_get():
    r = App.get('/')

    assert r.json()["result"] == 'Welcome to Adam Elshimis API.  Enter 14 set attributes of an individual to get a prediction on their salary'
    assert r.status_code == 200

def test_post_moreThan_correct():
    # Define the API URL
    api_url = "https://salary-prediction-fastapi.onrender.com/predict"      
    model = joblib.load("ml/model.joblib")
    encoder = joblib.load("ml/encoder.joblib")
    lb = joblib.load("ml/lb.joblib")

    moreThan50K = {  "age": 38,
    "workclass": "Federal-gov",
    "fnlgt": 125933,
    "education": "Masters",
    "education-num": 14,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "Iran"}

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

    r = requests.post(api_url, data=json.dumps(moreThan50K))

    assert r.json() == {'prediction': '>50K'}
    assert r.status_code != 200

def test_post_lessThan_correct():
    # Define the API URL
    api_url = "https://salary-prediction-fastapi.onrender.com/predict"
    model = joblib.load("ml/model.joblib")
    encoder = joblib.load("ml/encoder.joblib")
    lb = joblib.load("ml/lb.joblib")

    lessThan50K = {  "age": 28,
    "workclass": "Private",
    "fnlgt": 338409,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Wife",
    "race": "Black",
    "sex": "Female",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "Cuba"}

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

    r = requests.post(api_url, data=json.dumps(lessThan50K))

    assert r.json() == {'prediction': '<=50K'}
    assert r.status_code != 200