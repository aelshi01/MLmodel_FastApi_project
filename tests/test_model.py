#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 30

@author: adamelshimi
"""
# import sys
# sys.path.insert(0, '../')

from ml.model import train_model, compute_model_metrics, inference
from sklearn.model_selection import train_test_split
from ml.data import process_data
import pandas as pd
import numpy as np

    
def test_compute_model_metric():
    filename = './data/census.csv'
    # Add code to load in the data.
    data = pd.read_csv(filename,skipinitialspace = True)
    data.columns = data.columns.str.replace(' ', '')


    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.25)

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
        test, categorical_features=cat_features, label="salary", training=False,encoder= encoder, lb=lb
    )

    # Train and save a model.

    model = train_model(X_train, y_train)
    preds = model.predict(X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    assert 0 < precision < 1, "Precision should be between 0 and 1"
    assert 0 < recall < 1, "Recall should be between 0 and 1"
    assert 0 < fbeta < 1, "F1 Score should be between 0 and 1"

def test_inference():
    filename = './data/census.csv'
    # Add code to load in the data.
    data = pd.read_csv(filename,skipinitialspace = True)
    data.columns = data.columns.str.replace(' ', '')


    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.25)

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
        test, categorical_features=cat_features, label="salary", training=False,encoder= encoder, lb=lb
    )

    # Train and save a model.

    model = train_model(X_train, y_train)
    preds = inference(model,X_test)


    assert preds.shape == (8141,)

def test_train_model():
    filename = './data/census.csv'
    # Add code to load in the data.
    data = pd.read_csv(filename,skipinitialspace = True)
    data.columns = data.columns.str.replace(' ', '')


    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.25)

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
        test, categorical_features=cat_features, label="salary", training=False,encoder= encoder, lb=lb
    )

    # Train and save a model.

    model = train_model(X_train, y_train)

    assert model.n_features_in_ == 107
    # assert model.classes_ == np.array([0,1])
