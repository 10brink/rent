import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import os

def cluster_rentals(input_csv, output_csv, n_clusters=8):
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return

    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Filter out rows with missing latitude or longitude
    # Some rows have 'N/A' or might be empty
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    
    initial_count = len(df)
    df_clean = df.dropna(subset=['Latitude', 'Longitude']).copy()
    
    if len(df_clean) < n_clusters:
        print(f"Error: Not enough data points ({len(df_clean)}) to create {n_clusters} clusters.")
        return

    print(f"Clustering {len(df_clean)} records (dropped {initial_count - len(df_clean)} records with missing coordinates)...")

    # Extract coordinates for clustering
    coords = df_clean[['Latitude', 'Longitude']].values

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_clean['Cluster_ID'] = kmeans.fit_predict(coords)

    # Assign friendly names to neighborhoods based on cluster IDs
    df_clean['Neighborhood'] = df_clean['Cluster_ID'].apply(lambda x: f"Neighborhood {x + 1}")

    # Merge back to original dataframe if we want to keep the N/A rows (optional)
    # For now, let's just save the clustered results
    df_clean.to_csv(output_csv, index=False)
    print(f"Clustered data saved to {output_csv}")

    # Summary of clusters
    summary = df_clean.groupby('Neighborhood').size().reset_index(name='Count')
    centers = kmeans.cluster_centers_
    summary['Center_Lat'] = centers[:, 0]
    summary['Center_Long'] = centers[:, 1]
    
    print("\nCluster Summary:")
    print(summary.to_string(index=False))

if __name__ == "__main__":
    cluster_rentals('rentals_rentcast.csv', 'rentals_with_neighborhoods.csv')
