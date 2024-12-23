import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans
import seaborn as sns
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
def compute_cluster_stats(data):
    metrics=['Session count','session_duration','total_traffic (Bytes)']
    #group data with engagement group and compute statistics
    cluster_stats=data.groupby('engagement group')[metrics].agg(['min','max','mean','sum']).reset_index()
    
    return cluster_stats
def visualize_cluster_stats(data):
    #visualize clusters
    cluster_stats_melted = data.melt(id_vars='engagement group', var_name='Metric', value_name='Value')

    # Visualize metrics
    plt.figure(figsize=(16, 8))
    sns.barplot(data=cluster_stats_melted, x='engagement group', y='Value', hue='Metric')
    plt.title('Engagement Metrics by Cluster')
    plt.ylabel('Value')
    plt.xlabel('Engagement Group')
    plt.legend(title='metric')
    plt.show()
def application_traffic(data):
    # Define applications and their respective traffic columns
    apps = {
        "Netflix": ["Netflix DL (Bytes)", "Netflix UL (Bytes)"],
        "Youtube": ["Youtube DL (Bytes)", "Youtube UL (Bytes)"],
        "Gaming": ["Gaming DL (Bytes)", "Gaming UL (Bytes)"]
    }
    
    aggregated = {}
    pd_data=pd.DataFrame()
    
    # Aggregate the traffic for each application
    for app_name, cols in apps.items():
        # Calculate total traffic for the application
        data[f'{app_name} Total Traffic (Bytes)'] = data[cols].sum(axis=1)
        
        # Group by user and sum the total traffic
        aggregated[app_name] = (
            data.groupby("MSISDN/Number")[f'{app_name} Total Traffic (Bytes)']
            .sum()
            .reset_index()
            .sort_values(by=f'{app_name} Total Traffic (Bytes)', ascending=False)
        )

    pd_data = pd.concat(
        [df.assign(Application=app) for app, df in aggregated.items()],
        ignore_index=True
    )
    return pd_data
def top_10_users_per_app(data):
    # Define applications and their respective traffic columns
    apps = {
        "Netflix": [col for col in data.columns if "Netflix" in col],
        "Youtube": [col for col in data.columns if "Youtube" in col],
        "Gaming": [col for col in data.columns if "Gaming" in col],
    }

    top_users = {}

    for app_name, cols in apps.items():
        if not cols:
            print(f"Warning: No columns found for {app_name}")
            continue

        # Calculate total traffic for the application
        data[f'{app_name} Total Traffic (Bytes)'] = data[cols].sum(axis=1)

        # Group by user and sum traffic
        aggregated_data = (
            data.groupby("MSISDN/Number")[f'{app_name} Total Traffic (Bytes)']
            .sum()
            .reset_index()
        )

        # Sort by traffic and take top 10
        top_users[app_name] = aggregated_data.sort_values(
            by=f'{app_name} Total Traffic (Bytes)', ascending=False
        ).head(10)

    return top_users
def plot_top_3_apps(data):
    apps = {
        "Netflix": ["Netflix DL (Bytes)", "Netflix UL (Bytes)"],
        "Youtube": ["Youtube DL (Bytes)", "Youtube UL (Bytes)"],
        "Gaming": ["Gaming DL (Bytes)", "Gaming UL (Bytes)"]
    }
    
    # Calculate total traffic for each app
    total_traffic_app = {}
    for app_name, cols in apps.items():
        data[f"{app_name} Total traffic (bytes)"] = data[cols].sum(axis=1)
        
        # Sum the traffic for all users and store it in the dictionary
        total_traffic_app[app_name] = data[f"{app_name} Total traffic (bytes)"].sum()
    
    # Sort applications by total traffic in descending order and get the top 3
    sorted_apps = sorted(total_traffic_app.items(), key=lambda x: x[1], reverse=True)
    top_3_apps = sorted_apps[:3]
    
    # Extract app names and their traffic for plotting
    app_names, app_traffic = zip(*top_3_apps)
    
    # Plot the data
    plt.figure(figsize=(10,6))
    plt.bar(app_names, app_traffic, color=['blue', 'green', 'orange'])
    plt.title('Top 3 Most Used Applications')
    plt.xlabel('Application')
    plt.ylabel('Total Traffic (bytes)')
    plt.show()











