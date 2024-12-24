import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
def calculate_distance(data, centroid):
    return np.linalg.norm(data - centroid)
def engagement_scores(data):
    #extract relevant columns
    features=data[['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']]
    scaler=StandardScaler()
    data_scaled= scaler.fit_transform(features)
    #apply cluster
    kmeans=KMeans(n_clusters=3,random_state=0)
    data['cluster']= kmeans.fit_predict(data_scaled)
    centroids=kmeans.cluster_centers_
    # Calculate Engagement and Experience scores
    engagement_scores = []
    experience_scores = []
        # Extract user data (relevant features for the distance calculation)
    for index, row in data.iterrows():
        user_data = np.array([row['Avg RTT DL (ms)'], row['Avg RTT UL (ms)'], 
                              row['Avg Bearer TP DL (kbps)'], row['Avg Bearer TP UL (kbps)']])
        
        # Calculate the distance to the centroids of the relevant clusters
        engagement_score = calculate_distance(user_data, centroids[0])  # Cluster 0 centroid for Engagement
        experience_score = calculate_distance(user_data, centroids[2])  # Cluster 2 centroid for Experience
         # Append the scores to the lists
        engagement_scores.append(engagement_score)
        experience_scores.append(experience_score)
        # Add the scores to the DataFrame
    data['Engagement Score'] = engagement_scores
    data['Experience Score'] = experience_scores
    return data
def satisfaction_score(data):
    data['Satisfaction Score']=(data['Engagement Score']+data['Experience Score'])/2
    top_10_satisfied=data.sort_values(by='Satisfaction Score',ascending=False).head(10)
    return top_10_satisfied
def regression_model(data):
    #prepare the data
    features = ['Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']
    target = 'Satisfaction Score'
    x = data[features]
    y=data[target]
    #split test train
    X_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=42)
    #train the data
    model = XGBRegressor(n_estimators=200,random_state=42)
    #fit the mode
    model.fit(X_train,y_train)
    #make predictions on the test data
    prediction=model.predict(x_test)
    #test the accuracy
    mae_error=mean_absolute_error(prediction,y_test)
    return prediction,mae_error
def engagement_experience_cluster(data):
    # Select the relevant columns (Engagement Score & Experience Score)
    features = data[['Engagement Score', 'Experience Score']]

    # Standardize the data
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Apply K-means clustering with k=2
    kmeans = KMeans(n_clusters=2, random_state=0)
    data['eng_exp_Cluster'] = kmeans.fit_predict(features_scaled)

    return data
def aggregate_sat_exp(data):
    aggregated_score= data.groupby('eng_exp_Cluster')[['Satisfaction Score', 'Experience Score']].mean().reset_index()
    return aggregated_score