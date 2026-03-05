# %% Imports, options, and loading data
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.api import OLS
from statsmodels.formula.api import ols
import statsmodels.stats.outliers_influence

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# pandas display options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# matplotlib display options
plt.style.use('dark_background') # Matches the GitHub theme on my laptop better :)

# Load in data, remove white space, and encode categorical column
df_student = pd.read_csv("../data/Student_Performance.csv")
df_student.columns = df_student.columns.str.replace(' ', '', regex=False)

# Encode the categorical variable `ExtracurricularActivities` before train/test split
df_student = pd.get_dummies(df_student, columns=["ExtracurricularActivities"], drop_first=True) # `drop_first=True` because we're dealing with a simple binary category


# %% Data Preparation
# Split data into the features (independent variables) and target (dependent variable)
X, y = df_student.drop(columns=["PerformanceIndex"]), df_student["PerformanceIndex"]
print("X shape: ", X.shape)
print("y shape: ", y.shape, "\n")

# Convert and inspect data types
X = X.astype(float)
print("\nX dtypes:\n", X.dtypes, "\n")
print("y dtype: ", y.dtypes, "\n")


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

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Ensure data shapes are consistent
for name, var in [("X_train shape: ", X_train),
    ("X_test shape: ", X_test),
    ("y_train shape: ", y_train),
    ("y_test shape: ", y_test)]:
    print(name, var.shape)

# Add constant and create/fit the model
X_train, X_test = sm.add_constant(X_train), sm.add_constant(X_test)
print("X_train shape after constant: ", X_train.shape)
print("X_test shape after constant: ", X_test.shape)

sm_model = sm.OLS(y_train, X_train).fit() # Reminder that statsmodels uses y followed by X


# %% statsmodels OLS Results
# General information
print("Coefficients:\n", sm_model.params, "\n\n")
print(sm_model.summary(), "\n")

# Note: Will use sklearn MSE import later in the code
# Training data squared error and mean squared error
sm_train_pred = sm_model.predict(X_train)
sm_train_se = (sm_train_pred - y_train)**2
sm_train_mse = sm_train_se.mean()
print("Train Data Mean Squared Error: ", sm_train_mse)

# Test data squared error and mean squared error
sm_test_pred = sm_model.predict(X_test)
sm_test_se = (sm_test_pred - y_test)**2
sm_test_mse = sm_test_se.mean()
print("Test Data Mean Squared Error: ", sm_test_mse, "\n")

"""Also, here's the formula notation that you can use for exploring
different combinations of features. We will mess around with this
a bit after exploring `sm_model`.

This one uses only PreviousScores, for example:"""
example_formula_model = smf.ols(formula="PerformanceIndex ~ PreviousScores", data=df_student)
example_formula_model_results = example_formula_model.fit()
# %% R-squared and residual plot
"""
Now it's time to interpret these results. We'll begin by following this guide: https://medium.com/swlh/interpreting-linear-regression-through-statsmodels-summary-4796d359035a
in order to get a basic introduction on how to interpret the results.

Here's where you can find all the regression plots for statsmodels:
https://www.statsmodels.org/stable/graphics.html#regression-plots

From https://statisticsbyjim.com/regression/interpret-r-squared-regression/

'R-Squared indicated the percentage of the variance in the dependent variable
that the independent variables explain collectively. It measures the strength
of the relationship between your model and the dependent variable.

R-squared evaluates the scatter of the data points around the fitted
regression line. It's also called the coefficient of determination, or
the coefficient of multiple determination for multiple regression.

For the same data set, higher R-squared values represent smaller
differences between the observed data and the fitted values.

Limitations:
You cannot use R-squared to determine whether the coefficient estimates and
predictions are biased, which is why you must assess the residual plots.

R-squared does not indicate if a regression model provides an adequate fit
to your data. A good model can have a low R-squared value. On the other hand,
a biased model can have a high R-squared value.'
"""
sm_train_r2 = metrics.r2_score(y_train, sm_model.predict(X_train))
sm_test_r2 = metrics.r2_score(y_test, sm_model.predict(X_test))
print("R^2 Score (Training): ", sm_train_r2)
print("R^2 Score (Testing): ", sm_test_r2, "\n")

"""
Given the high R-squared values, we have some good first impressions on the
model, yet we should investigate further before making any conclusions.

The above article recommends residual plots for initial further exploration.
So... that's what we'll do
""";
# Get train and test model predictions
sm_train_predictions = sm_model.predict(X_train)
sm_test_predictions = sm_model.predict(X_test)

# Calculate train and test residuals
sm_train_resids = y_train - sm_train_predictions
sm_test_resids = y_test - sm_test_predictions

# Train data residual plot
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(sm_train_predictions, sm_train_resids)
ax.axhline(y=0, color='red', linewidth=1)

ax.set_xlabel("Predicted Values")
ax.set_ylabel("Residuals")
ax.set_title("Residual Plot (Training Data)")
plt.show();

# Test data residual plot
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(sm_test_predictions, sm_test_resids)
ax.axhline(y=0, color='red', linewidth=1)

ax.set_xlabel("Predicted Values")
ax.set_ylabel("Residuals")
ax.set_title("Residual Plot (Test Data)")
plt.show();
"""
Training Data Plot Analysis:
    It seems that at first glance the data generally lies around an error of plus or minus 2.5
    on predicting the PerformanceIndex. Let's calculate the mean absolute error to be sure of this
    before continuing.
"""
sm_train_mae = metrics.mean_absolute_error(y_train, sm_train_predictions)
sm_test_mae = metrics.mean_absolute_error(y_test, sm_test_predictions)
print("\nTrain MAE: ", sm_train_mae)
print("Test MAE: ", sm_test_mae)

# Let's also take a look at root MSE to see if there's any notable difference.
train_root_mse = np.sqrt(sm_train_mse)
test_root_mse = np.sqrt(sm_test_mse)
print("\nTrain Root MSE: ", train_root_mse)
print("Test Root MSE: ", test_root_mse)

"""
    Looks like the train and test MAE values were even lower than I thought after just looking
    at the graph. Another good sign that the model is doing well. Also, the root MSE values are
    also lower than my initial expectations based on the graph.

    Back to analyzing the training data plot. The final consideration I can think of here is
    the outliers. There doesn't seem to be any patterns indicating that the outliers are
    concentrated in any notable way. I feel like there's definitely a way to statistically
    confirm this, however.
""";

# %% Testing AIC and BIC with different feature combinations (for practice)

# %%
print("X_train shape: ", X_train.shape)

data = X_train.drop('const', axis=1)
print("data shape :", data.shape)
print("data type: ", type(data), "\n")

train_e = sm_train_resids - sm_train_pred
print("Train e:\n", train_e.head(4), "\n")

p = len(data.count(axis=0))
print("p data type: ", type(p))
print("p (parameters): ", p, "\n")

n = len(data)
print("n data type: ", type(n))
print("n (data points): ", "\n")

print(data.head())

#outliers = 3*(p/n)
#print("outliers: ", outliers)


# %% Defining Outliers
sm_outliers = statsmodels.stats.outliers_influence.OLSInfluence(sm_model)

"""
Before we begin with outliers, let's first cover the topic of
studentized residuals.

# From https://online.stat.psu.edu/stat462/node/247/
There are various measures for identifying extreme x values (high leverage observations), and
unusual y values (outliers). When trying to identify outliers, one problem that can arise is
is when there is a potential outlier that
"""

# Need to research the sigma parameter for being confident in this application
sm_studentized_resids = sm_outliers.get_resid_studentized_external(sigma=None)
print(sm_outliers)
