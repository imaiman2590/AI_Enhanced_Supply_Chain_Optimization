import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from preprocess import processdata

# Feature engineering
customer_data = processdata.groupby('customer_id').agg({
    'order_amount': ['mean', 'sum', 'count'],
    'date': lambda x: (pd.to_datetime(x).max() - pd.to_datetime(x).min()).days,
}).reset_index()

customer_data.columns = ['customer_id', 'avg_order', 'total_order', 'order_count', 'active_days']

# Normalize
features = customer_data.drop(columns='customer_id')
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)


kmeans = KMeans(n_clusters=4, random_state=42)
customer_df['kmeans_cluster'] = kmeans.fit_predict(X_scaled)

dbscan = DBSCAN(eps=1.2, min_samples=5)
customer_df['dbscan_cluster'] = dbscan.fit_predict(X_scaled)


