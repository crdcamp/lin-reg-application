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

# %% 2) Clean the Data Frame
"""
I don't believe this data needs to be cleaned, so we'll skip
this step for now. We'll save this for the harder challenge.
"""

# %% 3) Explore Global Properties.
"""
Use histograms, scatter plots, and
aggregation functions to summarize the data.
"""

"""
First, we'll test if there's any relationship between
hours of study and performance.

Then, we'll test if hours of sleep has an effect on performance.

Finally, we'll test if hours of study and sleep combined have an effect on performance.
"""

# Begin by testing number of unique hours

print()
