from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from ml.data import process_data

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    return lr

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.
    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return 'precision: {}'.format(precision), 'Recall: {}'.format(recall), 'fbeta: {}'.format(fbeta)


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def model_slices(model, feat: str=None):

    if feat:
        data = model[feat]
    else:
        data = model
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
        for cls in data['workclass'].unique():
            new_data = data[data["workclass"] == cls]
            train, test = train_test_split(new_data, test_size=0.25,random_state=99)
            
            try:
                X_train, y_train, encoder, lb = process_data(
                train, categorical_features=cat_features, label="salary", training=True
                )

                X_test, y_test, encoder, lb = process_data(
                test, categorical_features=cat_features, label="salary", training=False,encoder= encoder, lb=lb
                )

                model = train_model(X_train, y_train)
                preds = inference(model,X_test)
                model_metric = compute_model_metrics(y_test, preds)
                
                print(f'model_{cls} :', model_metric)

            except ValueError:
                print(f'model_{cls} Not valid data, target values contains only one class: 0')


