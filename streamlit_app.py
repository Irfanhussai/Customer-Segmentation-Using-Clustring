import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model_trainer import trained_objects
import matplotlib.cm as cm

# Streamlit Page Config
st.set_page_config(page_title="Customer Segmentation", layout="centered")

# Title
st.title("🧠 AI-Powered Customer Segmentation")
st.markdown("Enter customer details to predict their cluster using Autoencoder + KMeans.")

# Load trained objects
scaler = trained_objects["scaler"]
encoder = trained_objects["encoder"]
kmeans = trained_objects["kmeans"]
reducer = trained_objects["reducer"]
latent_features = trained_objects["latent_features"]
umap_features = trained_objects["umap_features"]
data = trained_objects["data"]
cluster_labels = trained_objects["cluster_labels"]

# Convert UMAP outputs to dense if not already
if not isinstance(umap_features, np.ndarray):
    umap_features = umap_features.toarray()

# Input fields
st.sidebar.header("Input Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 70, 30)
income = st.sidebar.slider("Annual Income (k$)", 10, 150, 60)
spending_score = st.sidebar.slider("Spending Score (1-100)", 1, 100, 50)

# Encode gender
gender_encoded = 1 if gender == "Male" else 0
sample = [gender_encoded, age, income, spending_score]

# Prediction Section
if st.sidebar.button("Predict Cluster"):
    sample_scaled = scaler.transform([sample])
    sample_latent = encoder.predict(sample_scaled)
    cluster = kmeans.predict(sample_latent)[0]
    cluster_name = cluster_labels.get(cluster, f"Cluster {cluster}")

    st.success(f"🧩 Predicted Cluster: **{cluster} - {cluster_name}**")

    # Update UMAP with the new sample
    all_latents = np.vstack([latent_features, sample_latent])
    umap_all = reducer.transform(all_latents)
    if not isinstance(umap_all, np.ndarray):
        umap_all = umap_all.toarray()

    fig, ax = plt.subplots(figsize=(10, 8))

    # Set custom colors
    palette = sns.color_palette("Set2", n_colors=len(np.unique(data["Cluster"])))

    # Plot existing points
    sns.scatterplot(
        x=umap_features[:, 0],
        y=umap_features[:, 1],
        hue=data["Cluster"].map(lambda x: cluster_labels.get(x, f"Cluster {x}")),
        palette=palette,
        alpha=0.7,
        s=80,
        edgecolor="k",
        linewidth=0.5,
        ax=ax
    )

    # Plot new user input
    ax.scatter(
        umap_all[-1, 0],
        umap_all[-1, 1],
        color='red',
        s=300,
        marker="X",
        edgecolor="black",
        linewidth=2,
        label="Your Input"
    )

    # Aesthetic Improvements
    ax.set_title("Customer Segmentation Map (UMAP Projection)", fontsize=18, fontweight='bold')
    ax.set_xlabel("UMAP Dimension 1", fontsize=14)
    ax.set_ylabel("UMAP Dimension 2", fontsize=14)
    ax.legend(title="Clusters", fontsize=10, title_fontsize=12, loc="best", bbox_to_anchor=(1.05, 1))
    sns.despine()
    plt.tight_layout()

    st.pyplot(fig)
