import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def data_extraction(data):
    data = pd.read_csv(data)
    return data
#cleaning the data to remove any
def cleaning_data(data):
    # Calculate the percentage of missing values per column
    count_percent = data.isnull().mean() * 100
    cleaned_data = data.copy()
    for column in data.columns:
        if count_percent[column] > 50:  # If more than 50% missing values, drop the column
            cleaned_data = cleaned_data.drop(columns=column)
        elif count_percent[column] <= 50:  # If less than or equal to 50% missing values
            if cleaned_data[column].dtype == 'object':  # For categorical data (object type)
                cleaned_data[column] = cleaned_data[column].fillna(cleaned_data[column].mode()[0])
            elif cleaned_data[column].dtype in ['int64', 'float64']:  # For numerical data
                cleaned_data[column] = cleaned_data[column].fillna(cleaned_data[column].mean())

    return cleaned_data
def remove_outliers(data):
    cleaned_data=data.copy()
    numeric_data = cleaned_data.select_dtypes(include=['number'])
    outliers=pd.DataFrame()
    for column in numeric_data.columns:
        q1=numeric_data[column].quantile(0.25)
        q3=numeric_data[column].quantile(0.75)
        IQR=q3-q1
        lower_bound=q1-1.5*IQR
        upper_bound=q3+1.5*IQR
        #identify outliers
        column_outliers=cleaned_data[(cleaned_data[column]<lower_bound) | (cleaned_data[column]>upper_bound)]
        #create and concatenate the outliers in a pd dataframe
        outliers=pd.concat([outliers,column_outliers])
        #clean outliers
        cleaned_data=cleaned_data[(cleaned_data[column]>= lower_bound)&(cleaned_data[column]<=upper_bound)]
    return cleaned_data,outliers