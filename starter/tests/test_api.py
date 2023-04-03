# import sys
# sys.path.insert(0, '../')

import pytest
from fastapi.testclient import TestClient
from src.mypkg.main import app

# comment
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


def test_get():
    r = App.get('/')

    assert r.json()["result"] == 'Welcome to Adam Elshimis API.  Enter 14 set attributes of an ndividual to get a prediction on their salary'
    assert r.status_code == 200

def test_post_moreThan_correct():
    r = App.post('/predict', json='moreThan50K')

    assert r.json()["salary"] == '>50K'
    assert r.status_code == 200