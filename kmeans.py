import sqlite3
import numpy as np

def load_data_from_sqlite(db_path, table_name):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")
    data = np.array(cursor.fetchall())
    conn.close()
    return data

def initialize_centroids(data, k):
    # Randomly choose k data points as initial centroids
    indices = np.random.choice(data.shape[0], k, replace=False)
    return data[indices]

def assign_clusters(data, centroids):
    distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
    return np.argmin(distances, axis=1)

def update_centroids(data, labels, k):
    new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids

def kmeans(data, k, max_iters=100):
    centroids = initialize_centroids(data, k)
    
    for _ in range(max_iters):
        labels = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, labels, k)
        
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    
    return labels, centroids

# Usage
if __name__ == "__main__":
    # Load data from SQLite database
    db_path = 'data.db'
    table_name = 'data_table'
    data = load_data_from_sqlite(db_path, table_name)

    # Choose the number of clusters
    k = 3  # Example: 3 clusters

    # Perform K-Means clustering
    labels, centroids = kmeans(data, k)

    print("Labels:", labels)
    print("Centroids:", centroids)
