import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
    aggregated_data['decile_class']=pd.qcut(aggregated_data['session_duration'],10,labels=False)+1#segments the users into 10 equal sized decile classes
    #group by decile class and calculate the total data per class
    aggregated_data=aggregated_data.groupby('decile_class').agg(total_data_per_decile=('total_data_volume','sum')).reset_index()
    top_decile=aggregated_data[aggregated_data['decile_class']>=5]#higher decile indicated higher data usage
    return top_decile#represents heavy users
def Non_graphical_univariate_analysis(data): #we will be using non graphical analysis
    #calculate the mean,median,mode,standard deviation, variance,range,min,max of the total data
    #created the dictionary to store the data and retain the name of the total data volume column
    dispersed_data={'Min':data['total_data_volume'].min(),
                    'Max':data['total_data_volume'].max(),
                    'Mean':data['total_data_volume'].mean(),
                    'Median':data['total_data_volume'].median(),
                    'Mode':data['total_data_volume'].mode(),
                    'Standard Deviation':data['total_data_volume'].std(),
                    'Variance':data['total_data_volume'].var(),
                    'Range':data['total_data_volume'].max()-data['total_data_volume'].min(),
                    }
    #convert the dictionary to a dataframe
    dispersed_data=pd.DataFrame(dispersed_data,index=[0])#used index=['0'] to create a single row dataframe
    return dispersed_data
def graphical_univariate_analysis(data):
    #plot and visualize the distribution,spread and central tendency of individual variables
    #using seaborn boxplot
    dispersed_data = {
    'Min': data['total_data_volume'].min(),
    'Max': data['total_data_volume'].max(),
    'Mean': data['total_data_volume'].mean(),
    'Median': data['total_data_volume'].median(),
    'Mode': data['total_data_volume'].mode()[0],  # Taking the first mode value
    'Standard Deviation': data['total_data_volume'].std(),
    'Variance': data['total_data_volume'].var(),
    'Range': data['total_data_volume'].max() - data['total_data_volume'].min()}
    # Create a bar plot
    plt.figure(figsize=(10,6))
    plt.bar(dispersed_data.keys(), dispersed_data.values(), color='skyblue', edgecolor='black')
    # Add labels and title
    plt.xlabel('Statistical Measures')
    plt.ylabel('Value')
    plt.title('Dispersion Parameters of Total Data Volume')
    # Show plot
    plt.show()
def correlation_analysis(data):
    #calculate the correlation between the total data volume and the other variables(social medias)
    applications=['total_youtube_dl',
       'total_youtube_ul', 'total_social_media_dl', 'total_social_media_ul',
       'total_gaming_dl', 'total_gaming_ul', 'total_email_dl',
       'total_email_ul', 'total_google_dl', 'total_google_ul',
       'total_other_dl', 'total_other_ul']
    #create a correlation matrix
    correlation_matrix=data[applications+['total_data_volume']].corr()
    #plot the correlation matrix
    return correlation_matrix
def PCA_analysis(data):
    #perform PCA analysis on the data
    #standardize the data
    features=['total_youtube_dl', 'total_youtube_ul', 'total_social_media_dl',
            'total_social_media_ul', 'total_gaming_dl', 'total_gaming_ul',
            'total_email_dl', 'total_email_ul', 'total_google_dl',
            'total_google_ul', 'total_other_dl', 'total_other_ul', 'total_data_volume']
    scale=StandardScaler()
    standardized_data=scale.fit_transform(data[features])
    #perform PCA
    pca = PCA(n_components=2)#retain 95% of the variance
    principal_components=pca.fit_transform(standardized_data)
    #convert the principal components to a dataframe
    principal_components=pd.DataFrame(principal_components,columns=['principal_component_1','principal_component_2'])
    return principal_components