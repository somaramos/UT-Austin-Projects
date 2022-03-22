import pandas as pd
import numpy as np

df_excel = pd.read_csv('StudentsPerformance.csv')

df_excel.describe()

# Calculate Average Math Scores
df_excel['math score'].mean()
