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
    # Data Engineer
    boston = datasets.load_boston()
    boston_df = pd.DataFrame(boston.data, columns=FEATURES)
    boston_df["LABELS"] = boston.target


    # Data Scientist
    lr = linear_model.LinearRegression()

    boston_df = boston_df[boston_df["LABELS"] < 50]

    predicted = cross_val_predict(lr, boston_df[FEATURES], boston_df["LABELS"], cv=10)
    results = mean_absolute_percentage_error(boston_df["LABELS"], predicted)
    return results


if __name__ == "__main__":
    results = run_model()
    print("Your MAPE Score is: {:.2f}%".format(results))
