# Link to data: https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression/data
# %% Imports and options
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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

# Basic Overview
print(df_student.head(), "\n")
print("Columns: \n", df_student.columns, "\n")
print("Shape: ", df_student.shape)
