# Link to data: https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression/data
# %% Imports and options
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

# pd.set_option("display.width", 800)
"""
EXPLORATORY DATA ANALYSIS

Simple EDA:

The TA from my beloved Harvard course recommends the following
ruberic for exploratory data analysis:
    1) Build a dataframe from the data (ideally, put all data in this object)
    2) Clean the dataframe: it should have the following properties:
        - Each row describes a single object
        - Each column describes a property of that object
        - Columns are numeric whenever appropriate
        - Columns contain atomic properties that cannot be further decomposed
    3) Explore global properties. Use histograms, scatter plots, and
       aggregation functions to summarize the data.
    4) Explore group properties. Use groupby and small multiples to
       compare subsets of the data.

So... that's what we'll do!
"""

# %% Load Data
df_student = pd.read_csv("data/Student_Performance.csv")

# %% 1) Build the Data Frame
# Basic Overview
print("Overview:\n", df_student.head(), "\n")
print("Shape: ", df_student.shape, "\n")
print("Description:\n", df_student.describe(), "\n")
print("Data Types:\n", df_student.dtypes, "\n")
print("Null Values:\n", df_student.isnull().sum())

# %% 2) Clean the Data Frame
"""
I don't believe this data needs to be cleaned, so we'll skip
this step for now. We'll save this for the challenge data.
"""

# %% 3) Explore Global Properties.
"""
Use histograms, scatter plots, and
aggregation functions to summarize the data.
"""

"""
1) Visualize and test a potential relationship between hours of study and performance.

2) Visualize and test a potential relationship of sleep on performance.

3) Visualize and test a potential relationship of both sleep and study on performance.
"""

# %% Histogram
# Define range of hours studied
print(
    "Range of hours studied:",
    df_student["Hours Studied"].max() - df_student["Hours Studied"].min(),
    "hours",
)

# Plot to visualize distribution
plt.hist(df_student["Hours Studied"], bins=9)
plt.xlabel("Hours of Study")
plt.ylabel("Student Count")

# Looks like the distribution of hours studied is roughly the same count across all hours.

# %% Scatter Plot
# Let's now check if there is any linear relationship when using a scatterplot.
sns.boxplot(
    data=df_student, x=df_student["Hours Studied"], y=df_student["Performance Index"]
);
# The results indicate that hours studied do have an effect on performance
