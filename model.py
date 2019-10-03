import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_predict

FEATURES = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B_RATIO",
    "LSTAT",
]


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def run_model():
    # Data
    boston = datasets.load_boston()
    boston_df = pd.DataFrame(boston.data, columns=FEATURES)
    boston_df["LABELS"] = boston.target

    # Model
    lr = linear_model.LinearRegression()
    predicted = cross_val_predict(lr, boston_df[FEATURES], boston_df["LABELS"], cv=10)
    return boston_df["LABELS"], predicted


if __name__ == "__main__":
    # Code
    labels, predicted = run_model()
    results = mean_absolute_percentage_error(labels, predicted)
    print("Your MAPE Score is: {:.2f}%".format(results))
