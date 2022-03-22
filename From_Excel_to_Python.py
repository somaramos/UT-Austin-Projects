import pandas as pd
import numpy as np

df_excel = pd.read_csv('StudentsPerformance.csv')

df_excel.describe()

# Calculate Average Math Scores
df_excel['math score'].mean()


df_excel['math score'].mean()
df_excel['math score'].max()
df_excel['math score'].min()
df_excel['math score'].count()

# Both of these add an  average row 
# df_excel['average'] = (df_excel['math score'] + df_excel['reading score'] + df_excel['writing score'])/3

df_excel['average'] = df_excel.mean(axis=1)

# Counts the Amount of Categorical Variables
df_excel['gender'].value_counts()

# IF 
# Replace IF with np.where()
df_excel['pass/fail'] = np.where(df_excel['average'] > 70, 'Pass', 'Fail')  #looks for an average of better than 

# for multiple conditions 
# make a list of conditions 
conditions = [
    (df_excel['average']>=90),
    (df_excel['average']>=80) & (df_excel['average']<90),
    (df_excel['average']>=70) & (df_excel['average']<80),
    (df_excel['average']>=60) & (df_excel['average']<70),
    (df_excel['average']>=50) & (df_excel['average']<60),
    (df_excel['average']<50),
]

# make a list of values A-F
values = ['A', 'B', 'C', 'D', 'E', 'F']

# make a grades colum
df_excel['grades'] = np.select(conditions, values)

# check the new data frame 
df_excel.describe()


# Conditionals in Python 
df_female = df_excel[df_excel['gender'] == 'female'] # this makes a new data frame of only females

# Mulitple Conditionals
df_sumifs = df_excel[(df_excel['gender'] == 'female') & (df_excel['race/ethnicity'] == 'group B')]

# some more calculations with df_sumifs
df_sumifs = df_sumifs.assign(sumifs = df_sumifs['math score'] + df_sumifs['reading score'] + df_sumifs['writing score'])


#*********
#
# BASIC DATA CLEANING 
#

df_excel['gender'].str.title()
df_excel['gender'].str.upper()
df_excel['gender'].str.title()

# this saves the values 
df_excel['gender'] = df_excel['gender'].str.title()


# looking for empty cells 
df_excel[df_excel['gender'].isnull()]

#Vlookup 
excel_1 = 'StudentsPerformance.csv'
excel_2 = 'LanguageScore.csv'
df_excel_1 = pd.read_csv(excel_1)
df_excel_2 = pd.read_csv(excel_2)

df_excel_1 = df_excel_1.reset_index()
df_excel_1 = df_excel_1.rename(columns={'index':'id'})


df_excel_1.loc[100, ] # row 100

df_excel_1.loc[df_excel_1['id']==100, 'math score']  # math score of row 100

# combine tables 
df_excel_3 = pd.merge(df_excel_1, df_excel_2, on='id', how='left')
df_excel_3['language score'].fillna('0', inplace=True)


# combine using concat
df_excel_3 = pd.concat(
 [df_excel_1.set_index('id'), df_excel_2.set_index('id')], axis=1
)
df_excel_3['language score'].fillna('0', inplace=True)
df_excel_3

# pivot_table
df_excel = pd.read_csv('StudentsPerformance.csv')
df_pivot = df_excel.pivot_table(index='race/ethnicity', values=['math score', 'writing score'], aggfunc='mean')
df_pivot


# ******
# graph time!
import matplotlib.pyplot as plt

#barplot
df_plot = df_pivot.reset_index()
plt.bar(df_plot['race/ethnicity'], df_plot['math score'])
plt.show()
