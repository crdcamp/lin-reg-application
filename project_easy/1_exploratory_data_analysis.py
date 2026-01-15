# %% Imports and options
# Data Source: https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression/data
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing

plt.style.use('dark_background')
#pd.set_option("display.width", 800)
"""
====================================================================================================
EXPLORATORY DATA ANALYSIS
====================================================================================================

Simple EDA:

The TA from my beloved Harvard course recommends the following
rubric for exploratory data analysis:
(Source: )
    1) Build a data frame from the data (ideally, put all data in this object)
    2) Clean the dataframe: it should have the following properties:
        - Each row describes a single object
        - Each column describes a property of that object
        - Columns are numeric whenever appropriate
        - Columns contain atomic properties that cannot be further decomposed
    3) Explore global properties. Use histograms, scatter plots, and
       aggregation functions to summarize the data.
    4) Explore group properties. Use `groupby` and small multiples to
       compare subsets of the data.

So... that's what we'll do! (Where applicable)
""";

# %% Load Data
df_student = pd.read_csv("../data/Student_Performance.csv")

# %% 1) Build the Data Frame (Replacing this with some introductory investigation)
# Basic Overview
print("Overview:\n", df_student.head(), "\n")
print("Shape: ", df_student.shape, "\n")
print("Description:\n", df_student.describe(), "\n")
print("Data Types:\n", df_student.dtypes, "\n")
print("Null Values:\n", df_student.isnull().sum())
# Let's change `Extracurricular Activities` to bool


# %% 2) Clean the Data Frame
"""
I don't believe this data needs to be cleaned, so we'll skip
this step for now. We'll save this for the challenge data.
""";

# %% 3) Explore Global Properties.
"""
Use histograms, scatter plots, and
aggregation functions to summarize the data.
""";

"""
INITIAL DATA EXPLORATION:

1) Visualize and test a potential relationship between hours of study and performance.

2) Visualize and test a potential relationship of sleep on performance.

3) Visualize and test a potential relationship of both sleep and study on performance.

4) Visualize all the data to explore other potential correlations and relationships

(Yes, I know the plotting should be functions but I want to practice)
""";

"""HOURS STUDIED VS. PERFORMANCE""";
# %% Hours Studied vs. Performance Histogram
# Define range of hours studied
print(
    "Range of hours studied:",
    df_student["Hours Studied"].max() - df_student["Hours Studied"].min(),
    "hours",
)

print("Max hours of study: ", df_student["Hours Studied"].max())
print("Min hours of study: ", df_student["Hours Studied"].min())

# Plot to visualize distributions for hours of study
plt.figure()
sns.countplot(data=df_student, x="Hours Studied", order=df_student["Hours Studied"].value_counts().index)
plt.show();
"""
Looks like the distribution of hours studied is roughly the same count across all hours.
However, it's worth noting that most students studied one hour (kinda surprising).
""";

# %% Hours Studied vs. Performance Box Plot
# Let's now check if there is any linear relationship when using a scatter plot.
plt.figure()
sns.boxplot(
    data=df_student, x="Hours Studied", y="Performance Index"
)
plt.show();
"""
The results indicate that hours studied has a positive impact on performance
This conclusion is further reinforced by the fact that the distribution of the
student count and the hours of study is (unrealistically) even.

Moreover, the boxplot spread (is that the correct terminology?) indicates
that the spread of scores based on hours of study is also pretty even
""";

# %% Hours Studied vs. Performance Violin Plot
# Now let's get an idea of the distribution density using a violin plot
plt.figure()
sns.violinplot(
    data=df_student, x=df_student["Hours Studied"], y=df_student["Performance Index"], split=True
)
plt.show();

"""
Again, the distributions are unusually even (likely due to this being an introductory data set).
With my limited knowledge, I believe this further reinforces that hours studied might be a
strong predictor for performance
""";

"""SLEEP VS. PERFORMANCE""";
# %% Sleep vs. Performance Histogram
print("Range of sleep hours: ", df_student["Sleep Hours"].max() - df_student["Sleep Hours"].min())
print("Max hours of sleep: ", df_student["Sleep Hours"].max())
print("Min hours of sleep: ", df_student["Sleep Hours"].min(), "\n")

plt.figure()
sns.countplot(data=df_student, x="Sleep Hours", order=df_student["Sleep Hours"].value_counts().index)
plt.show();
"""
Same story as before. We got a pretty even distribution here as well. However,
most students slept 8 hours.
""";

# %% Sleep vs. Performance Box Plot
plt.figure()
sns.boxplot(data=df_student, x="Sleep Hours", y="Performance Index")
plt.show();
"""
Judging by initial impressions from this boxplot, it seems that sleep probably
doesn't have as much of an effect on performance as hours of study.
""";

# %% Sleep vs. Performance Violin Plot
plt.figure()
sns.violinplot(data=df_student, x="Sleep Hours", y="Performance Index", split=True)
plt.show();
"""
Again, not seeing much of a trend here. There may be some minor
implications for 4 hours of sleep (People either get 40% or 70%),
but I don't think this is indicative of any trend as of now.
""";

# %% Visualize and test a potential relationship of both sleep and study on performance
# First, we'll need to split the data and make a multiple linear regression model
train_data, test_data = train_test_split(df_student, test_size=0.2, random_state=42)

X_train, X_test = train_data[["Hours Studied", "Sleep Hours"]], test_data[["Hours Studied", "Sleep Hours"]]
y_train, y_test = train_data["Performance Index"], test_data["Performance Index"]

print("X_train and y_train shape:", X_train.shape, y_train.shape)

study_sleep_model = LinearRegression()
study_sleep_model.fit(X_train, y_train)

sleep_study_predictions = study_sleep_model.predict(X_test)
