# Link to data: https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression/data
# %% Imports and options
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
EXPLORATORY DATA ANALYSIS


"""

# %% Load Data
df_student = pd.read_csv("data/Student_Performance.csv")

print(df_student.head(), "\n")
print("Rows: ", len(df_student), "\n")
print("Columns: \n", df_student.columns, "\n")
print("Shape: ", df_student.shape)
