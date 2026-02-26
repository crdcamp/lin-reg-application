# %% Imports, options, and original data
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.style.use('dark_background') # Matches the GitHub theme on my laptop better :)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# No need for data cleaning since we're working with easy data that doesn't require it
df_student = pd.read_csv("../data/Student_Performance.csv")

# %% Data Preparation
# Split data into the features (independent variables) and target (dependent variable)
X, y = df_student.drop(columns=["Performance Index"]), df_student["Performance Index"]
print("X shape: ", X.shape)
print("y shape: ", y.shape, "\n")

# Encode the categorical variable `Extracurricular Activities`
X = pd.get_dummies(X, columns=["Extracurricular Activities"], drop_first=True)

# Split into train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("X_train, X_test shape: ", X_train.shape, X_test.shape)
print("y_train, y_test shape: ", y_train.shape, y_test.shape, "\n")

# Inspect data types before continuing (just in case)
print("X dtypes:\n", X.dtypes, "\n")
print("y dtype:\n", y.dtypes, "\n")

# %% statsmodels Model Creation
"""
Now it's time to start creating some models!

We'll begin with a multiple linear regression model using all variables in the predictions
using the statsmodels library.

Then, we'll thoroughly inspect the data with all the tools statsmodels provides in order
to brush up on interpreting linear regression models.
""";
