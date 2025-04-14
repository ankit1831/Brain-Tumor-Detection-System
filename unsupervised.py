import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'data\Brain Tumor.csv')
df_numeric = df[['Energy', 'Homogeneity']].dropna()
X = StandardScaler().fit_transform(df_numeric)
X_train, X_test = train_test_split(X, test_size=0.33, random_state=42)
X_train_pca = PCA(n_components=2).fit_transform(X_train)

def elbow_method(X, max_k=10):
    wcss = [KMeans(n_clusters=k, init="k-means++", random_state=42).fit(X).inertia_ for k in range(1, max_k+1)]
    plt.plot(range(1, 11), wcss, marker='o')
    plt.xlabel("Number of Clusters (K)"), plt.ylabel("WCSS"), plt.title("Elbow Method"), plt.show()

def cluster_and_plot(model, X, X_pca, title):
    labels = model.fit_predict(X)
    score = silhouette_score(X, labels) if len(set(labels)) > 1 else -1
    print(f"{title} Silhouette Score: {score:.3f}")
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='plasma', marker='o')
    plt.title(title), plt.show()

elbow_method(X_train)

cluster_and_plot(KMeans(n_clusters=2), X_train, X_train_pca, "K-Means Clustering")

plt.figure(figsize=(8, 6))
dendrogram(linkage(X_train, method='ward'))
from sklearn.cluster import AgglomerativeClustering
plt.title("Hierarchical Clustering Dendrogram"), plt.xlabel("Data Points"), plt.ylabel("Distance"), plt.show()
cluster_and_plot(AgglomerativeClustering(n_clusters=4, linkage='ward'), X_train, X_train_pca, "Hierarchical Clustering")


dbscan_labels = DBSCAN(eps=0.1,min_samples=10).fit_predict(X_train)

plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=dbscan_labels, cmap="viridis")
plt.title("DBSCAN Clustering ")
plt.show()

# Anomaly Detection using Isolation Forest and LOF
def anomaly_detection(X, model, title):
    labels = model.fit_predict(X)
    
    plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=labels, cmap="coolwarm", marker='o')
    plt.title(title), plt.show()

anomaly_detection(X_train, IsolationForest(contamination=0.03), "Isolation Forest Anomalies")
anomaly_detection(X_train, LocalOutlierFactor(n_neighbors=10, contamination=0.02), "LOF Anomalies")
