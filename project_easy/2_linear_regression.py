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
# Actually... let's remove the white space:
df_student.columns = df_student.columns.str.replace(' ', '', regex=False)

# %% Data Preparation
# Split data into the features (independent variables) and target (dependent variable)
X, y = df_student.drop(columns=["PerformanceIndex"]), df_student["PerformanceIndex"]
print("X shape: ", X.shape)
print("y shape: ", y.shape, "\n")

# Encode the categorical variable `ExtracurricularActivities` before train/test split
X = pd.get_dummies(X, columns=["ExtracurricularActivities"], drop_first=True)
X = X.astype(float)

# Ensure data shapes are consistent
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

Then, we'll test out different variations of a multiple linear regression model with statsmodels
and further test potential relationships between features. For instance, in `1_exploratory_data_analysis.py`,
we were only testing for Pearson (linear) correlations. We want to test for other possible correlations as well.

Then, we'll thoroughly inspect the data with all the tools statsmodels provides in order
to brush up on interpreting linear regression models.
""";

# Add a constant and split the data
X = sm.add_constant(X)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Note: X variables have a different shape due to the addition of the constant.")
for name, var in [("X_train shape: ", X_train),
    ("X_test shape: ", X_test),
    ("y_train shape: ", y_train),
    ("y_test shape: ", y_test)]:
    print(name, var.shape)
print()

sm_model = sm.OLS(y_train, X_train) # Reminder that statsmodels uses Y followed by X
sm_results = sm_model.fit()

# %% Interpreting statsmodels OLS Results
# General information
print("\nSTATSMODELS MODEL RESULTS")
print("Coefficients:\n", sm_results.params, "\n")
print(sm_results.summary(), "\n")

# Train vs. test data information
# Also, who would've thought... the sklearn metrics are compatible with the statsmodels results!
print("R^2 Score (Training): ", metrics.r2_score(y_train, sm_results.predict(X_train)))
print("R^2 Score (Testing): ", metrics.r2_score(y_test, sm_results.predict(X_test)), "\n")

# Note: Will use sklearn MSE import later in the code

# Training data squared error and mean squared error
sm_model_train_pred = sm_results.predict(X_train)
sm_model_train_se = (sm_model_train_pred - y_train)**2
sm_model_train_mse = sm_model_train_se.mean()
print("Train Data Mean Squared Error: ", sm_model_train_mse)

# Test data squared error and mean squared error
sm_model_test_pred = sm_results.predict(X_test)
sm_model_test_se = (sm_model_test_pred - y_test)**2
sm_model_test_mse = sm_model_test_se.mean()
print("Test Data Mean Squared Error: ", sm_model_test_mse)

"""Also, here's the formula notation that you can use for exploring
different combinations of features. We will mess around with this
a bit after exploring `sm_results`.

This one uses only PreviousScores, for example:"""
example_formula_model = smf.ols(formula="PerformanceIndex ~ PreviousScores", data=df_student)
example_formula_model_results = example_formula_model.fit()

# %% Plots for Initial Model Results
"""
Now it's time to interpret these results. We'll begin by graphing them
and checking out those results.

After that, let's type some notes for all these results from the summary we're about to review.
While the data we're working with here is incredibly simple, we also
have to keep in mind that these will be important in a more advanced application.
""";
# Residuals plot
fig = plt.figure(figsize=(12, 8))
fig = sm.graphics.plot_regress_exog(sm_results, 'PreviousScores', fig=fig)
plt.show()

# Actual vs. predicted plot

# %% Interpreting the Summary Results

# %% Testing AIC and BIC with different feature combinations (for practice)
