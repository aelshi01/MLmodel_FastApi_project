# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics
from sklearn.preprocessing import OneHotEncoder, label_binarize, LabelBinarizer

filename = './data/census.csv'
# Add code to load in the data.
data = pd.read_csv(filename,skipinitialspace = True)
data.columns = data.columns.str.replace(' ', '')


# Optional enhancement, use K-fold cross validation instead of a train-test split.
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

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, encoder, lb = process_data(
    test, categorical_features=cat_features, label="salary", training=True
)

# Train and save a model.


model = train_model(X_train, y_train)
pred = inference(model,X_test)
# model = ...  # Get model (Sequential, Functional Model, or Model subclass)
# model.save('./path/')

# y = data.pop('salary')
# X = data

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23)

# ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
# X_train = ohe.fit_transform(X_train.values)
# X_test = ohe.transform(X_test.values)

# lb = LabelBinarizer()
# y_train = label_binarize(y_train.values,classes=['>50k','<=50k']).ravel()
# y_test = label_binarize(y_test.values,classes=['>50k','<=50k']).ravel()

# lr = LogisticRegression(max_iter=1000)
# lr.fit(X_train, y_train)

# scores = lr.predict_proba(X_test)
# pred = lr.predict(X_test)

# # F1 = 2 * (precision * recall) / (precision + recall)
# f1 = f1_score(y_test, pred)


if __name__=='__main__':
    #print(compute_model_metrics(y_test, pred))
    #print(data['workclass'])
    #print(X_train, y_train, encoder, lb)
    # print(test.__sizeof__)
    # print('train')
    # print(train.values.shape)
    # print('test')
    # print(test.values.shape)
    # print('X_train - after process data')
    # print(X_train.shape)
    # print('X_test - after process data')
    # print(X_test.shape)
    #print(f"F1 score: {f1:.4f}")

    print(pred)