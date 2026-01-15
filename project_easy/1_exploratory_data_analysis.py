# %% Imports and options
# Data Source: https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression/data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

So... that's what we'll do!
""";

# %% Load Data
df_student = pd.read_csv("../data/Student_Performance.csv")

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
""";

# %% 3) Explore Global Properties.
"""
Use histograms, scatter plots, and
aggregation functions to summarize the data.
""";

"""
Initial data exploration:

1) Visualize and test a potential relationship between hours of study and performance.

2) Visualize and test a potential relationship of sleep on performance.

3) Visualize and test a potential relationship of both sleep and study on performance.

4) Visualize all the data to explore other potential correlations and relationships
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
plt.xlabel("Hours of Study")
plt.ylabel("Student Count")
plt.hist(df_student["Hours Studied"], bins=9)
plt.show();
"""
Looks like the distribution of hours studied is roughly the same count across all hours.
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

# %% Hours Studied vs. Performance Scatter Plot
# Then let's throw in a scatterplot just to see what happens (I think the box and violin plots already tell us enough though,
# and I don't believe a scatterplot applies to the categorical nature of the hours anyway).
plt.figure()
hours_sorted = df_student["Hours Studied"].sort_values()
performance_sorted = df_student["Performance Index"].sort_values()
sns.regplot(data=df_student, x=hours_sorted, y=performance_sorted)
plt.show();
"""
Seems to be consistent with the histograms. Also worth noting that both 1 hour
of study and 9 hours of study have greater differences between the lowest and
highest performers relative to the rest of the study hour sets.
This was also shown in the violin plot as well.
""";

"""SLEEP VS. PERFORMANCE""";
# %% Sleep vs. Performance Histogram
print("Range of sleep hours: ", df_student["Sleep Hours"].max() - df_student["Sleep Hours"].min())
print("Max hours of sleep: ", df_student["Sleep Hours"].max())
print("Min hours of sleep: ", df_student["Sleep Hours"].min(), "\n")

plt.figure()
plt.xlabel("Hours of Sleep")
plt.ylabel("Student Count")
plt.hist(df_student["Sleep Hours"], bins=6);
"""
Same story as before. We got a pretty even distribution here as well.
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

# %% Sleep vs. Performance Scatter Plot

plt.figure()
sns.regplot(data=df_student, x="Sleep Hours", y="Performance Index")
plt.show();
"""
Yep. There doesn't seem to be any real usable trend here.
"""
