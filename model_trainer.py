import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import umap

# Load dataset
data = pd.read_csv("Mall_Customers.csv")
X = data.copy()

# Encode gender
X["Gender"] = LabelEncoder().fit_transform(X["Gender"])  # Male:1, Female:0

# Keep only necessary columns
feature_columns = ["Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)"]
X = X[feature_columns]

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)

# Autoencoder definition
def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    latent = Dense(20, activation='relu')(encoded)  # Increased latent dimension
    decoded = Dense(32, activation='relu')(latent)
    decoded = Dense(64, activation='relu')(decoded)
    output = Dense(input_dim, activation='linear')(decoded)

    autoencoder = Model(inputs=input_layer, outputs=output)
    encoder = Model(inputs=input_layer, outputs=latent)

    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

# Build and train the autoencoder
autoencoder, encoder = build_autoencoder(X_scaled.shape[1])
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
autoencoder.fit(
    X_train, X_train,
    epochs=200,
    batch_size=32,
    validation_data=(X_val, X_val),
    callbacks=[early_stop],
    verbose=0
)

# Extract latent features
latent_features = encoder.predict(X_scaled)

# KMeans clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(latent_features)

# Check cluster distribution for sanity
unique, counts = np.unique(clusters, return_counts=True)
print("Cluster distribution:", dict(zip(unique, counts)))
data["Cluster"] = clusters

# UMAP projection for visualization
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42, transform_mode='graph')
umap_features = reducer.fit_transform(latent_features)
# Ensure dense array for downstream plotting
if not isinstance(umap_features, np.ndarray):
    umap_features = umap_features.toarray()

# Manual cluster labels
cluster_labels = {
    0: "Cautious Elites",
    1: "Impulsive Spenders",
    2: "Budget Conscious",
    3: "Young & Social",
    4: "High-Value Loyalists"
}

# Export trained objects
trained_objects = {
    "scaler": scaler,
    "encoder": encoder,
    "kmeans": kmeans,
    "reducer": reducer,
    "latent_features": latent_features,
    "umap_features": umap_features,
    "data": data,
    "cluster_labels": cluster_labels,
    "feature_columns": feature_columns
}
