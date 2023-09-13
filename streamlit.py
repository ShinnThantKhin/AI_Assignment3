import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

# Load the dataset from the provided link
data = pd.read_csv("mobile_price/train.csv")

# Remove the "price_range" column as we are performing clustering
X = data.drop("price_range", axis=1)

# Preprocess the data: Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA for dimensionality reduction (optional)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Determine the number of clusters using the Elbow Method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Sidebar
st.sidebar.title("Cluster Visualization")

# Main content
st.title("Mobile Clustering Visualization")

# Display the Elbow Method graph
st.subheader("Elbow Method for Optimal Cluster Number")
st.pyplot(plt.figure(figsize=(8, 6)))
plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia (Within-Cluster Sum of Squares)')
plt.title('Elbow Method for Optimal Cluster Number')

# User input for selecting the number of clusters
num_clusters = st.sidebar.slider("Select the Number of Clusters", 2, 10, 4)

# Apply K-Means clustering with the chosen number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Visualize the clusters using PCA (you can choose different visualization techniques)
st.subheader("Clustering of Mobile Devices (PCA)")
plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')
plt.title('Clustering of Mobile Devices (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
st.pyplot()

# Explore the clusters and their characteristics
cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=X.columns)

# Display cluster centers
st.subheader("Cluster Centers")
st.write(cluster_centers)

# You can further analyze and interpret the clusters based on their characteristics

# Save the cluster labels to a CSV file if needed
data['cluster'] = clusters
data.to_csv('mobile_clusters.csv', index=False)
