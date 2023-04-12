
import requests
import json

# Define the API URL
api_url = "https://salary-prediction-fastapi.onrender.com/predict"

# Define the sample input data
input_data = {  "age": 38,
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

# Make a POST request to the API
response = requests.post(api_url, data=json.dumps(input_data))

# Print the status code and the result of the model inference
print("Status code:", response.status_code)
print("Prediction:", response.json())