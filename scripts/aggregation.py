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