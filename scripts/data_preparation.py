import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def data_extraction(data):
    data=pd.read_csv(data)
    return data
#this function calculates the percentage of missing data in each column
def get_missing_data_percentage(data):
    count = data.isnull().sum()
    count_percent = (count/len(data))*100
    return count_percent

def data_cleaning(data):#clean data from missing values
    count_percent = get_missing_data_percentage(data)#calling get_missing_data_percentage function
    #so here we are calculating the mode of the data since we have calculated the number of missing data
    if (count_percent>50).any():#if the misssing data of the column is more than 50% drop the column
        data=data.drop(columns=count_percent[count_percent>50].index)
    else:#if the missing data is less than 50% impute the missing data with the mode of the data
        for column in data.columns:
            if data[column].dtype=='object':
                data[column]=data[column].fillna(data[column].mode()[0])#we use [0] because there may me more than one mode
            elif data[column].dtype=='int64' or data[column].dtype=='float64':
                data[column]=data[column].fillna(data[column].mean())
    #count_2=data.isnull().sum()
    return data
def identify_outliers(data):
    #this function identifies the outliers in the data and removes them
    #data_outlined=data_cleaning(data)#calling data_cleaning function to prepare the data
    numeric_data=data.select_dtypes(include=['int64','float64'])
    q1=numeric_data.quantile(0.25)#we use the quantile function in pandas to calculate the 25th percentile
    q3=numeric_data.quantile(0.75)#we use the 75th percentile
    IQR = q3-q1#we calculate the interquartile
    lower_bound = q1-1.5*IQR#we calculate the lower bound and use 1.5 number because it is the most common number used to identify outliers
    upper_bound = q3+1.5*IQR#we calculate the upper bound
    data_mean = numeric_data.mean() #mean of each column
    modified_data = data.copy()
    for column in numeric_data.columns:
        # Replace outliers with mean for each column
        mask = (numeric_data[column] < lower_bound[column]) | (numeric_data[column] > upper_bound[column])
        modified_data.loc[mask, column] = data_mean[column]#mask is a boolean series with the same length as number of rows
    return modified_data
