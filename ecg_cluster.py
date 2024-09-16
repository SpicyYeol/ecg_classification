from analysis import load_and_preprocess_data
from preprocess_utils import segment_ecg
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from itertools import chain

N_DATA = [2] # 완료 2,4,6
OFFSET = None
DEBUG = False
LEARNING_RATE = 0.001
PLOT = False
dtype = 1
CLUSTERING = True

def main():
    preprocessed_dataset = load_and_preprocess_data(OFFSET, N_DATA,dtype,DEBUG,CLUSTERING,PLOT)
    for item in preprocessed_dataset:
        item['data'] = np.asarray(segment_ecg(item['data']))[:,:,0] # ECG 데이터를 분할

    data_combined = np.vstack([entry['data'] for entry in preprocessed_dataset])
    labels_combined = list(chain(*[[entry['label']] * len(entry['data']) for entry in preprocessed_dataset]))

    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data_combined)

    def combined_clustering_analysis():
        cluster_range = range(2, 10)
        sse = []  # Sum of Squared Errors
        silhouette_scores = []  # Silhouette Scores

        for n_clusters in cluster_range:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(data_combined)

            sse.append(kmeans.inertia_)
            silhouette_avg = silhouette_score(data_combined, kmeans.labels_)
            silhouette_scores.append(silhouette_avg)

        # Plot Elbow and Silhouette Score
        fig, ax1 = plt.subplots(figsize=(8, 6))

        ax1.set_xlabel('Number of Clusters')
        ax1.set_ylabel('SSE', color='tab:blue')
        ax1.plot(cluster_range, sse, marker='o', color='tab:blue', label="SSE")
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Silhouette Score', color='tab:green')
        ax2.plot(cluster_range, silhouette_scores, marker='o', color='tab:green', label="Silhouette Score")
        ax2.tick_params(axis='y', labelcolor='tab:green')

        fig.tight_layout()
        plt.title("Elbow Method and Silhouette Score")
        plt.show()

    # Step 4: DBSCAN Clustering and Visualization
    def dbscan_clustering():
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(data_combined)

        # Check how many clusters DBSCAN found (-1 represents noise)
        unique_labels_dbscan = np.unique(dbscan_labels)
        n_clusters_dbscan = len(unique_labels_dbscan) - (1 if -1 in unique_labels_dbscan else 0)

        # Visualize DBSCAN results
        plt.figure(figsize=(8, 6))
        scatter_dbscan = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=dbscan_labels, cmap='viridis')

        # Annotate points with their labels
        for i, label in enumerate(labels_combined):
            plt.annotate(label, (data_2d[i, 0], data_2d[i, 1]), fontsize=9)

        plt.title(f"DBSCAN Clustering Visualization (Estimated Clusters: {n_clusters_dbscan})")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.show()
    print("combined_clustering")
    combined_clustering_analysis()  # Elbow Method and Silhouette Score analysis
    print("dbscan_clustering")
    dbscan_clustering()  # DBSCAN clustering and visualization


if __name__ == "__main__":
    main()
