
import requests
import json

# Define the API URL
api_url = "https://salary-prediction-fastapi.onrender.com/predict/"

# Define the sample input data
input_data = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education_num": 13,
    "marital_status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital_gain": 2174,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": "United-States",
}

# Make a POST request to the API
response = requests.post(api_url, data=json.dumps(input_data))

# Print the status code and the result of the model inference
print("Status code:", response.status_code)
print("Prediction:", response.json())