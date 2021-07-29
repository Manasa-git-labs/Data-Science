#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np


ser1 = pd.Series([1.5, 2.5, 3, 4.5, 5.0, 6])
print(ser1)

 
ser2 = pd.Series(["India", "Canada", "Germany"], name="Countries")
print(ser2)


ser4 = pd.Series({"India": "New Delhi",
                  "Japan": "Tokyo",
                  "UK": "London"})
print(ser4)


values = ["India", "Canada", "Australia",
          "Japan", "Germany", "France"]
 
code = ["IND", "CAN", "AUS", "JAP", "GER", "FRA"]
 
ser1 = pd.Series(values, index=code)
 
print("-----Head()-----")
print(ser1.head())
 
print("\n\n-----Head(2)-----")
print(ser1.head(2))

print("\n\n-----Tail(2)-----")
print(ser1.tail(2))

print("-----Take()-----")
print(ser1.take([2, 4, 5]))

##Adding new column to existing DataFrame

employees = pd.DataFrame({
    'EmpCode': ['Emp001', 'Emp002', 'Emp003', 'Emp004', 'Emp005'],
    'Name': ['John', 'Doe', 'William', 'Spark', 'Mark'],
    'Occupation': ['Chemist', 'Statistician', 'Statistician',
                   'Statistician', 'Programmer'],
    'Date Of Join': ['2018-01-25', '2018-01-26', '2018-01-26', '2018-02-26',
                     '2018-03-16'],
    'Age': [23, 24, 34, 29, 40]})
 
employees['City'] = ['London', 'Tokyo', 'Sydney', 'London', 'Toronto']
 
print(employees)


##Select multiple columns from DataFrame


df = employees[['EmpCode', 'Age', 'Name']]
print(df)


##Get mean(average) of rows and columns
 
df = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [5, 5, 0, 0]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3'])
 
df['Mean Basket'] = df.mean(axis=1)
df.loc['Mean Fruit'] = df.mean()
 
print(df)

##Calculate sum across rows and columns

df['Sum Basket'] = df.sum(axis=1)
df.loc['Sum Fruit'] = df.sum()
 
print(df)


##handling missing data

df = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [5,]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3'])
 
print("\n--------- DataFrame ---------\n")
print(df)
 
print("\n--------- Use of isnull() ---------\n")
print(df.isnull())
 
print("\n--------- Use of notnull() ---------\n")
print(df.notnull())


#count distinct values

df = pd.DataFrame({'Age': [30, 20, 22, 40, 20, 30, 20, 25],
                    'Height': [165, 70, 120, 80, 162, 72, 124, 81],
                    'Score': [4.6, 8.3, 9.0, 3.3, 4, 8, 9, 3],
                    'State': ['NY', 'TX', 'FL', 'AL', 'NY', 'TX', 'FL', 'AL']},
                   index=['Jane', 'Nick', 'Aaron', 'Penelope', 'Jaane', 'Nicky', 'Armour', 'Ponting'])
 
print(df.Age.value_counts())


#get values of a specific cell
df = pd.DataFrame({'Age': [30, 40, 30, 40, 30, 30, 20, 25],
                   'Height': [120, 162, 120, 120, 120, 72, 120, 81],
                   'Score': [4.6, 4.6, 9.0, 3.3, 4, 8, 9, 3],
                   'State': ['NY', 'NY', 'FL', 'AL', 'NY', 'TX', 'FL', 'AL']},
                  index=['Jane', 'Jane', 'Aaron', 'Penelope', 'Jaane', 'Nicky',
                         'Armour', 'Ponting'])
 
print(df.loc['Nicky', 'Age'])



#delete missing rows
df = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [5,]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3'])
 
print("\n--------- DataFrame ---------\n")
print(df)
 
print("\n--------- Use of dropna() ---------\n")
print(df.dropna())


#drop columns with missing data
df = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [5,]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3'])
 
print("\n--------- DataFrame ---------\n")
print(df)
 
print("\n--------- Drop Columns) ---------\n")
print(df.dropna(1))


#Sort Column in descending order
employees = pd.DataFrame({
    'EmpCode': ['Emp001', 'Emp002', 'Emp003', 'Emp004', 'Emp005'],
    'Name': ['John', 'Doe', 'William', 'Spark', 'Mark'],
    'Occupation': ['Chemist', 'Statistician', 'Statistician',
                   'Statistician', 'Programmer'],
    'Date Of Join': ['2018-01-25', '2018-01-26', '2018-01-26', '2018-02-26',
                     '2018-03-16'],
    'Age': [23, 24, 34, 29, 40]})
 
 
print(employees.sort_index(axis=1, ascending=False))


#Finding minimum and maximum values
df = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [55, 15, 8, 12],
                   [15, 14, 1, 8], [7, 1, 1, 8], [5, 4, 9, 2]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3', 'Basket4',
                         'Basket5', 'Basket6'])
 
print("\n----------- Minimum -----------\n")
print(df[['Apple', 'Orange', 'Banana', 'Pear']].min())
 
print("\n----------- Maximum -----------\n")
print(df[['Apple', 'Orange', 'Banana', 'Pear']].max())



#Summary statistics

df = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [55, 15, 8, 12],
                   [15, 14, 1, 8], [7, 1, 1, 8], [5, 4, 9, 2]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3', 'Basket4',
                         'Basket5', 'Basket6'])
 
print("\n----------- Describe DataFrame -----------\n")
print(df.describe())
 
print("\n----------- Describe Column -----------\n")
print(df[['Apple']].describe())


#Find Mean, Median and Mode
df = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [55, 15, 8, 12],
                   [15, 14, 1, 8], [7, 1, 1, 8], [5, 4, 9, 2]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3', 'Basket4',
                         'Basket5', 'Basket6'])
 
print("\n----------- Calculate Mean -----------\n")
print(df.mean())
 
print("\n----------- Calculate Median -----------\n")
print(df.median())
 
print("\n----------- Calculate Mode -----------\n")
print(df.mode())


#Measure Variance and Standard Deviation
print(df.std())
print(df.var())
print(df['Apple'].var())


#Calculating Covariance
df = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [55, 15, 8, 12],
                   [15, 14, 1, 8], [7, 1, 1, 8], [5, 4, 9, 2]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3', 'Basket4',
                         'Basket5', 'Basket6'])
 
print("\n----------- Calculating Covariance -----------\n")
print(df.cov())
 
print("\n----------- Between 2 columns -----------\n")
# Covariance of Apple vs Orange
print(df.Apple.cov(df.Orange))


#Calculating correlation between two DataFrame

df1 = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [55, 15, 8, 12],
                   [15, 14, 1, 8], [7, 1, 1, 8], [5, 4, 9, 2]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3', 'Basket4',
                         'Basket5', 'Basket6'])
 
print("\n------ Calculating Correlation of one DataFrame Columns -----\n")
print(df1.corr())
 
df2 = pd.DataFrame([[52, 54, 58, 41], [14, 24, 51, 78], [55, 15, 8, 12],
                   [15, 14, 1, 8], [7, 17, 18, 98], [15, 34, 29, 52]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3', 'Basket4',
                         'Basket5', 'Basket6'])
 
print("\n----- Calculating correlation between two DataFrame -------\n")
print(df2.corrwith(other=df1))


#Forward and backward filling of missing values
df = pd.DataFrame([[10, 30, 40], [], [15, 8, 12],[15, 14, 1, 8], [7, 8], [5, 4, 1]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'], index=['Basket1', 'Basket2', 'Basket3', 'Basket4',
                         'Basket5', 'Basket6'])
 
print("\n------ DataFrame with NaN -----\n")
print(df)
 
print("\n------ DataFrame with Forward Filling -----\n")
print(df.ffill())
 
print("\n------ DataFrame with backward Filling -----\n")
print(df.bfill())

print("\n------ DataFrame with backward Filling -----\n")
print(df.fillna(value=df.mean()))

print(df.mean())

      




# In[12]:


import pandas as pd
import numpy as np
df=pd.read_csv("Indianstates.csv")
print(df.head())
print(df.tail())
print(df.mean())
print(df.population.mean())

