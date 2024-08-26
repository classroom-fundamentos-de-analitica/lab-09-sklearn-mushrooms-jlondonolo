"""GitHub Classroom autograding script."""

import os
import pickle

import pandas as pd
from sklearn.metrics import accuracy_score


def load_estimator():
    """Load trained model from disk."""

    if not os.path.exists("model.pkl"):
        return None
    with open("model.pkl", "rb") as file:
        estimator = pickle.load(file)

    return estimator


def load_datasets():
    """Load train and test datasets."""

    train_dataset = pd.read_csv("train_dataset.csv")
    test_dataset = pd.read_csv("test_dataset.csv")

    train_dataset['type'] = train_dataset['type'].map({'p': 1, 'e': 0})
    test_dataset['type'] = test_dataset['type'].map({'p': 1, 'e': 0})

    from sklearn.preprocessing import LabelEncoder

    # Create a LabelEncoder object
    label_encoder = LabelEncoder()

    for col in train_dataset.columns:
        if col != 'type':
            # Encode the column using LabelEncoder
            train_dataset[col] = label_encoder.fit_transform(train_dataset[col])
            test_dataset[col] = label_encoder.transform(test_dataset[col])

    x_train = train_dataset.drop("type", axis=1)
    y_train = train_dataset["type"]

    x_test = test_dataset.drop("type", axis=1)
    y_test = test_dataset["type"]

    return x_train, x_test, y_train, y_test


def eval_metrics(y_true, y_pred):
    """Evaluate model performance."""

    accuracy = accuracy_score(y_true, y_pred)

    return accuracy


def compute_metrics():
    """Compute model metrics."""

    estimator = load_estimator()
    assert estimator is not None, "Model not found"

x_train, x_test, y_true_train, y_true_test = load_datasets()
estimator = load_estimator()
y_pred_train = estimator.predict(x_train)
y_pred_test = estimator.predict(x_test)

accuracy_train = eval_metrics(y_true_train, y_pred_train)
accuracy_test = eval_metrics(y_true_test, y_pred_test)

def test_():
    """Run grading script."""
    assert accuracy_train > 0.99
    assert accuracy_test > 0.99

test_()


