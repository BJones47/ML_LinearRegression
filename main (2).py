

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# Define the linear regression function
def linear_regression(X_train, y_train):
    # Add a column of ones for the intercept term
    X_train = np.c_[np.ones(X_train.shape[0]), X_train]

    # Calculate coefficients using the normal equation
    coeffs = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)

    return coeffs

# Define the function to calculate RMSE
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

# Read in auto-mpg.data file 
df = pd.read_csv('auto-mpg.data', header=None, sep='\s+', na_values="?")

# Drop rows with missing values '?'
df = df.dropna()

# Split data into Coefficients (X) dependent variable (y)
y = df[0]
X = df.iloc[:, 1:8]

# Define number of folds for cross-validation
num_folds = 10
kf = KFold(n_splits=num_folds)

# Initialize lists to store coefficients and RMSE values for each fold
coefficients = []
rmse_values = []

# Perform 10-fold cross-validation
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Perform linear regression
    coeffs = linear_regression(X_train, y_train)
    coefficients.append(coeffs[1:])  # Exclude the intercept term

    # Make predictions
    y_pred = np.dot(np.c_[np.ones(X_test.shape[0]), X_test], coeffs)

    # Calculate RMSE
    rmse_val = rmse(y_pred, y_test)
    rmse_values.append(rmse_val)

# Display coefficients and RMSE values in a tabular format
col_width = 12
print("Fold".ljust(col_width), "|", "Cylinders".ljust(col_width), "|", "Displacement".ljust(col_width), "|", 
      "Horsepower".ljust(col_width), "|", "Weight".ljust(col_width), "|", "Acceleration".ljust(col_width), "|", 
      "Model Year".ljust(col_width), "|", "Origin".ljust(col_width), "| RMSE")
print("-" * (col_width * 7 + 45))
for i in range(num_folds):
    print(f"{i + 1}".ljust(col_width), "|", end='')
    coeffs_str = " | ".join([f"{coeff:.6f}".ljust(col_width) for coeff in coefficients[i]])
    print(coeffs_str, "|", rmse_values[i])
    print('\n')
