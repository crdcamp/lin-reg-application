# %% Imports, options, and original data
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.api import OLS

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

# Encode the categorical variable `Extracurricular Activities` before train/test split
X = pd.get_dummies(X, columns=["Extracurricular Activities"], drop_first=True)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Split into train and test data
for name, var in [("X_train shape: ", X_train),
    ("X_test shape: ", X_test),
    ("y_train shape: ", y_train),
    ("y_test shape: ", y_test)]:
    print(name, var.shape)

# Inspect data types
print("\nX dtypes:\n", X.dtypes, "\n")
print("y dtype:\n", y.dtypes, "\n")

# %% statsmodels Model Creation
"""
Now it's time to start creating some models!

We'll begin with a multiple linear regression model using all variables in the predictions
using the statsmodels library.

Then, we'll thoroughly inspect the data with all the tools statsmodels provides in order
to brush up on interpreting linear regression models.

Before we continue though, let's make all of the previously created variables into numpy data types
to make statsmodels's extremely inflexible data type handling happy.

We'll assign them separately to avoid confusion later and keep the feature and target variable
column names for the scikit application.
""";

X_np = X.astype(float)
y_np = y.astype(float)
X_np, y_np = X_np.to_numpy(), y_np.to_numpy()

X_np = sm.add_constant(X_np)

# Do a separate split for the converted values
X_np_train, X_np_test, y_np_train, y_np_test = train_test_split(
    X_np, y_np, test_size=0.2, random_state=42
)

for name, var in [("X_np_train shape: ", X_np_train),
    ("X_np_test shape: ", X_np_test),
    ("y_np_train shape: ", y_np_train),
    ("y_np_test shape: ", y_np_test)]:
    print(name, var.shape)
print()

# Add constant because for some reason the creator of statsmodels didn't make this a default behavior
sm_model = sm.OLS(y_np_train, X_np_train) # Reminder that statsmodels using Y followed by X
sm_results = sm_model.fit()
print("Statsmodels model coefficients:\n", sm_results.params, "\n") # Display the coefficients of the fitted model
