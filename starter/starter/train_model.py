# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
from sklearn.preprocessing import OneHotEncoder, label_binarize, LabelBinarizer
from joblib import dump
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

filename = './data/census.csv'
# Add code to load in the data.
logger.info("reading data file")
data = pd.read_csv(filename,skipinitialspace = True)
data.columns = data.columns.str.replace(' ', '')


# Optional enhancement, use K-fold cross validation instead of a train-test split.
logger.info('splitting data train and test')
# Split arrays or matrices into random train and test subsets.
train, test = train_test_split(data, test_size=0.25,random_state=99)

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

logger.info('process train data using OneHotEncoder and LabelBinarizer.')
# Process the data used in the machine learning pipeline.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
logger.info('process test data using OneHotEncoder and LabelBinarizer.')
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=False,encoder= encoder, lb=lb
)

# Train and save a model.

logger.info('training model with logistic regression')
model = train_model(X_train, y_train)

logger.info('saving model')
dump(model, "./starter/ml/model.joblib")
logger.info('saving encoder')
dump(encoder, "./starter/ml/encoder.joblib")
logger.info('saving LabelBinarizer')
dump(lb, "./starter/ml/lb.joblib")

logger.info('All process completed SUCCESSFULY!')
