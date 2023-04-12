from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from ml.data import process_data
from sklearn.model_selection import train_test_split

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


def model_slices(model, data,feature, cat_features: str, encoder,lb):

    results = {}
    for cls in data[feature].unique():
        new_data = data[data[feature] == cls]
        
        try:
            _, test = train_test_split(new_data, test_size=0.25,random_state=99)
            
            X_test, y_test, _, _ = process_data(
            test, categorical_features=cat_features, label="salary", training=False,encoder= encoder, lb=lb
            )
            preds = inference(model,X_test)
            model_metric = compute_model_metrics(y_test, preds)
            results[cls] = model_metric

        except ValueError:
            results[cls] = f'slice_{cls} Not valid data, too little data or only has one class'
    
    return results

