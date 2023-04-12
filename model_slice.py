import joblib
from ml.model import model_slices
import json

# Load the model from disk
model = joblib.load("ml/model.joblib")
encoder = joblib.load("ml/encoder.joblib")
lb = joblib.load("ml/lb.joblib")

filename = './data/census.csv'
# Add code to load in the data.
data = pd.read_csv(filename,skipinitialspace = True)
data.columns = data.columns.str.replace(' ', '')

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

results = model_slices(model, data,  "native-country",cat_features,encoder,lb)

with open('./slice_performance.txt', 'w') as out:
    out.write(json.dumps(results, indent = 4))