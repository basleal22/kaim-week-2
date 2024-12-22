import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans
def __init__(self,data):
    self.data=data
    return data
def aggregation_function(data):
    session_frequency=data.groupby('MSISDN/Number').size().reset_index(name='Session count')
    session_duration=data.groupby('MSISDN/Number')['Dur. (ms)'].sum().reset_index(name='session_duration')
    session_trafic=data.groupby('MSISDN/Number').agg({'Total UL (Bytes)':'sum',
                                                      'Total DL (Bytes)':'sum'}).reset_index()
    aggregated_data=session_frequency.merge(session_duration,on='MSISDN/Number').merge(session_trafic,on='MSISDN/Number')
    top_10_sessions_freq=aggregated_data.sort_values('Session count',ascending=False).head(10)
    top_10_sessions_dur=aggregated_data.sort_values('session_duration',ascending=False).head(10)
    aggregated_data['total_traffic (Bytes)']=aggregated_data['Total UL (Bytes)'] + aggregated_data['Total DL (Bytes)']

    top_10_traffic = aggregated_data.sort_values('total_traffic (Bytes)',ascending=False).head(10)
     #   ''
    #})
    return top_10_sessions_freq,top_10_sessions_dur,top_10_traffic,aggregated_data
def kmean_normalized(data):
    metrics=['Session count','session_duration','total_traffic (Bytes)']
    if data[metrics].isnull().any().any():
        raise ValueError("There are missing values in the data.")
    scaler=StandardScaler()
    
    scaled_data=scaler.fit_transform(data[metrics])
    #create new pd dataframe with the metrics in the above and the datas being normalized
    scaled_df = pd.DataFrame(scaled_data, columns=metrics)
    #apply k-mean clustering 
    kmeans=KMeans(n_clusters=3,random_state=42)
    data['engagement group']=kmeans.fit_predict(scaled_df)
    sorted_dur = data.sort_values('session_duration',ascending=False)
    return data,sorted_dur

