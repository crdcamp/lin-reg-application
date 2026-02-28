# %% Imports, options, and original data
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.api import OLS

from sklearn import metrics
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

X_np, y_np = X.astype(float), y.astype(float)
X_np, y_np = X_np.to_numpy(), y_np.to_numpy()

X_np = sm.add_constant(X_np)

# Do a separate split for the converted values
X_np_train, X_np_test, y_np_train, y_np_test = train_test_split(
    X_np, y_np, test_size=0.2, random_state=42
)

print("Note: X_np variables have a different shape due to the addition of the constant.")
for name, var in [("X_np_train shape: ", X_np_train),
    ("X_np_test shape: ", X_np_test),
    ("y_np_train shape: ", y_np_train),
    ("y_np_test shape: ", y_np_test)]:
    print(name, var.shape)
print()

sm_model = sm.OLS(y_np_train, X_np_train) # Reminder that statsmodels uses Y followed by X
sm_results = sm_model.fit()

# %% Interpreting statsmodels OLS Results
# General information
print("\nSTATSMODELS MODEL RESULTS")
print("Coefficients:\n", sm_results.params, "\n")
print(sm_results.summary(), "\n")

# Train vs. test data information
# Also, who would've thought... the sklearn metrics are compatible with the statsmodels results!
print("R^2 Score (Training): ", metrics.r2_score(y_np_train, sm_results.predict(X_np_train)))
print("R^2 Score (Testing): ", metrics.r2_score(y_np_test, sm_results.predict(X_np_test)), "\n")

# Note: Will use sklearn MSE import later in the code

# Training data squared error and mean squared error
sm_model_train_pred = sm_results.predict(X_np_train)
sm_model_train_se = (sm_model_train_pred - y_np_train)**2
sm_model_train_mse = sm_model_train_se.mean()
print("Train Data Mean Squared Error: ", sm_model_train_mse)

# Test data squared error and mean squared error
sm_model_test_pred = sm_results.predict(X_np_test)
sm_model_test_se = (sm_model_test_pred - y_np_test)**2
sm_model_test_mse = sm_model_test_se.mean()
print("Test Data Mean Squared Error: ", sm_model_test_mse)

"""Also, here's the formula notation that you can use for exploring
different combinations of features.
This one uses only Previous Scores, for example:"""

# Need to remove white space so the smf formula will accept the inputs
df_smf_example = df_student.copy()
df_smf_example.columns = df_smf_example.columns.str.replace(' ', '', regex=False)

example_formula_model = smf.ols(formula="PerformanceIndex ~ PreviousScores", data=df_smf_example)

# %% Interpreting the results
"""
Now it's time to interpret these results. HOWEVER, before we do so      ,
let's type some notes for all these factors we're about to review.
While the data we're working with here is incredibly simple, we also
have to keep in mind that these will be important in a more advanced application.
""";

# %% Testing AIC and BIC with different feature combinations (for practice)
