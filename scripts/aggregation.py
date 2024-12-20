import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def data_aggregation(data):
    aggregated_data=data.groupby('IMSI').agg(number_of_sessions=('Bearer Id','count'),
                                         session_duration=('Dur. (ms)','sum'),
                                         total_youtube_dl=('Youtube UL (Bytes)','sum'),
                                         total_youtube_ul=('Youtube DL (Bytes)','sum'),
                                         total_social_media_dl=('Social Media DL (Bytes)','sum'),
                                         total_social_media_ul=('Social Media UL (Bytes)','sum'),
                                         total_gaming_dl=('Gaming DL (Bytes)','sum'),
                                         total_gaming_ul=('Gaming UL (Bytes)','sum'),
                                         total_email_dl=('Email DL (Bytes)','sum'),
                                         total_email_ul=('Email UL (Bytes)','sum'),
                                         total_google_dl=('Google DL (Bytes)','sum'),
                                         total_google_ul=('Google UL (Bytes)','sum'),
                                         total_other_dl=('Other DL (Bytes)','sum'),
                                         total_other_ul=('Other UL (Bytes)','sum'),
                                         ).reset_index()
        #summing the total data volume
    aggregated_data['total_download']=aggregated_data[['total_youtube_dl','total_social_media_dl','total_gaming_dl','total_email_dl','total_google_dl','total_other_dl']].sum(axis=1)
    aggregated_data['total_upload']=aggregated_data[['total_youtube_ul','total_social_media_ul','total_gaming_ul','total_email_ul','total_google_ul','total_other_ul']].sum(axis=1)
    aggregated_data['total_data_volume']=aggregated_data[['total_download','total_upload']].sum(axis=1)
    return aggregated_data
def segment_data(data):
    #sort the data by session duration
    aggregated_data=data.sort_values(by='session_duration',ascending=False)
    #create a new column for decile class
    aggregated_data['decile_class']=pd.qcut(aggregated_data['session_duration'],10,label=False)+1#segments the users into 10 equal sized decile classes
    #calculate total volume of upload and download for each class and create a new column for it
    aggregated_data['total_data']=aggregated_data[['total_download','total_upload']].sum(axis=1)#sum(axis=1) because axis =1 is for rows
    #group by decile class and calculate the total data per class
    aggregated_data=aggregated_data.groupby('decile_class').agg(total_data_per_decile=('total_data','sum')).reset_index()
    return aggregated_data