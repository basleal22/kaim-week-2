import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
def aggregated(data):
    numeric_aggregated= data.groupby("MSISDN/Number").agg({"Avg RTT DL (ms)":"mean",
                                                       "Avg RTT UL (ms)":"mean",
                                                       "Avg Bearer TP DL (kbps)": "mean",
                                                       "Avg Bearer TP UL (kbps)": "mean"})
                                                       
    #find the most common handset type per msisdn
    handset= data.groupby("MSISDN/Number")['Handset Type'].agg(pd.Series.mode).reset_index()
    merged_data=pd.merge(numeric_aggregated, handset, on="MSISDN/Number")
    return merged_data
def top_10_values(data):
    # Define relevant columns
    columns = {
           'RTT': ['Avg RTT DL (ms)', 'Avg RTT UL (ms)'],
            'Throughput': ['Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']
                }

        # Create dictionaries to store results
    top_10 = {}
    bottom_10 = {}
    most_frequent = {}

    # Loop through each metric group
    for metric, cols in columns.items():
        top_10[metric] = {}
        bottom_10[metric] = {}
        most_frequent[metric] = {}
        for col in cols:
         # Compute top 10, bottom 10, and most frequent values
            top_10[metric][col] = data[col].nlargest(10).tolist()
            bottom_10[metric][col] = data[col].nsmallest(10).tolist()
            most_frequent[metric][col] = data[col].value_counts().head(10).index.tolist()
    return top_10,bottom_10,most_frequent
def distribution_visualize(data):
    import seaborn as sns

    # Group by Handset Type and calculate metrics
    grouped_data = data.groupby('Handset Type').agg({
        'Avg Bearer TP DL (kbps)': 'mean',
        'Avg Bearer TP UL (kbps)': 'mean', 
                        }).reset_index()

    # Rename columns for clarity
    grouped_data.rename(columns={
        'Avg Bearer TP DL (kbps)': 'Avg_Throughput_DL',
        'Avg Bearer TP UL (kbps)': 'Avg_Throughput_UL'}, inplace=True)
    
    # Sort data by Download Throughput
    sorted_data = grouped_data.sort_values(by='Avg_Throughput_DL', ascending=False).head(20)    

    # Plot Download Throughput (Top 20 Handsets)
    plt.figure(figsize=(12, 8))
    sns.barplot(
    x='Avg_Throughput_DL', 
    y='Handset Type', 
    data=sorted_data, 
    palette="viridis")
    plt.title('Top 20 Handset Types by Average Download Throughput')
    plt.xlabel('Average Download Throughput (kbps)')
    plt.ylabel('Handset Type')
    plt.tight_layout()
    plt.show()

    # Plot Upload Throughput (Top 20 Handsets)
    plt.figure(figsize=(12, 8))
    sns.barplot(
    x='Avg_Throughput_UL', 
    y='Handset Type', 
    data=sorted_data, 
    palette="magma"
)
    plt.title('Top 20 Handset Types by Average Upload Throughput')
    plt.xlabel('Average Upload Throughput (kbps)')
    plt.ylabel('Handset Type')
    plt.tight_layout()
    plt.show()

    #group data by RTT
    data['Avg RTT']=(data['Avg RTT DL (ms)']+data['Avg RTT UL (ms)'])/2
    #compute the average RTT value(upload and download together)
    avg_rtt_data = data.sort_values(by='Avg RTT', ascending=False).head(20)
    #plot the top 20 handset types by average RTT
    plt.figure(figsize=(10,6))
    sns.barplot(x='Avg RTT',
        y='Handset Type',
        data=avg_rtt_data,
        palette="magma")
    plt.title('Top 20 Handset Types by RTT')
    plt.xlabel('RTT upload and download')
    plt.ylabel('Handset Type')
    plt.tight_layout()
    plt.show()
def performance_clusters(data):
    # Rename columns for clarity
    data.rename(columns={
        'Avg Bearer TP DL (kbps)': 'Avg_Throughput_DL',
        'Avg Bearer TP UL (kbps)': 'Avg_Throughput_UL'}, inplace=True)
    # Ensure Avg RTT is computed
    data['Avg RTT'] = (data['Avg RTT DL (ms)'] + data['Avg RTT UL (ms)']) / 2

    # Select relevant features
    features = data[['Avg_Throughput_DL', 'Avg_Throughput_UL', 'Avg RTT']]

    # Normalize the data
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(features)

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['Cluster'] = kmeans.fit_predict(normalized_features)

    # Summarize each cluster
    cluster_summary = data.groupby('Cluster')[['Avg_Throughput_DL', 'Avg_Throughput_UL', 'Avg RTT']].mean()

    return cluster_summary


   
