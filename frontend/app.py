"""Interactive Streamlit frontend for the DBSCAN clustering project.

Run from the project root:
    streamlit run frontend/app.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


st.set_page_config(
    page_title="DBSCAN Clustering Dashboard",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 DBSCAN Clustering Dashboard")
st.write(
    "Explore density-based clustering, tune DBSCAN parameters, "
    "identify noise points, and evaluate the clustering results."
)

# Sidebar controls
st.sidebar.header("Dataset and DBSCAN settings")
n_samples = st.sidebar.slider("Number of samples", 150, 1500, 750, step=50)
cluster_std = st.sidebar.slider(
    "Cluster spread (cluster_std)", 0.1, 1.2, 0.4, step=0.05
)
eps = st.sidebar.slider("Epsilon (eps)", 0.05, 1.50, 0.30, step=0.05)
min_samples = st.sidebar.slider(
    "Minimum samples (min_samples)", 2, 30, 10, step=1
)
random_state = st.sidebar.number_input(
    "Random state", min_value=0, max_value=9999, value=0, step=1
)

st.sidebar.caption(
    "Tip: Increasing eps generally connects more points. "
    "Increasing min_samples makes the core-point density requirement stricter."
)

# Generate and scale data
centers = [[1, 1], [-1, -1], [1, -1]]
X_raw, labels_true = make_blobs(
    n_samples=n_samples,
    centers=centers,
    cluster_std=cluster_std,
    random_state=int(random_state),
)
X = StandardScaler().fit_transform(X_raw)

# Fit DBSCAN
model = DBSCAN(eps=eps, min_samples=min_samples)
labels = model.fit_predict(X)

core_mask = np.zeros_like(labels, dtype=bool)
core_mask[model.core_sample_indices_] = True

unique_labels = set(labels)
n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
n_noise = int(np.sum(labels == -1))
noise_pct = (n_noise / len(labels)) * 100

# Summary metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Clusters found", n_clusters)
col2.metric("Noise points", n_noise)
col3.metric("Noise percentage", f"{noise_pct:.1f}%")
col4.metric("Total samples", len(labels))

st.divider()

# Cluster visualization
left, right = st.columns([1.6, 1])

with left:
    st.subheader("DBSCAN clustering result")
    fig, ax = plt.subplots(figsize=(9, 6))

    # Plot each discovered cluster. Noise is black.
    cluster_ids = sorted(unique_labels)
    cmap = plt.cm.get_cmap("Spectral", max(len(cluster_ids), 1))

    for color_index, cluster_id in enumerate(cluster_ids):
        member_mask = labels == cluster_id
        color = (0, 0, 0, 1) if cluster_id == -1 else cmap(color_index)

        # Core samples are larger.
        core_points = X[member_mask & core_mask]
        if len(core_points):
            ax.scatter(
                core_points[:, 0],
                core_points[:, 1],
                c=[color],
                s=75,
                edgecolors="black",
                linewidths=0.5,
                label=f"Cluster {cluster_id} - core" if cluster_id != -1 else "Noise - core",
            )

        # Non-core samples are smaller (border points, or noise).
        other_points = X[member_mask & ~core_mask]
        if len(other_points):
            ax.scatter(
                other_points[:, 0],
                other_points[:, 1],
                c=[color],
                s=24,
                edgecolors="black",
                linewidths=0.4,
                marker="o",
                label=f"Cluster {cluster_id} - border" if cluster_id != -1 else "Noise",
            )

    ax.set_xlabel("Scaled feature 1")
    ax.set_ylabel("Scaled feature 2")
    ax.set_title(f"DBSCAN result (eps={eps:.2f}, min_samples={min_samples})")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, loc="best")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with right:
    st.subheader("Point types")
    core_count = int(np.sum(core_mask))
    border_count = int(np.sum((labels != -1) & ~core_mask))

    point_type_df = pd.DataFrame(
        {
            "Point type": ["Core points", "Border points", "Noise points"],
            "Count": [core_count, border_count, n_noise],
        }
    )
    st.dataframe(point_type_df, hide_index=True, use_container_width=True)

    st.info(
        "**Core:** has at least `min_samples` points in its eps-neighborhood "
        "(including itself).\n\n"
        "**Border:** is not a core point but is within the neighborhood of "
        "a core point.\n\n"
        "**Noise:** is not assigned to any cluster."
    )

st.divider()

# Evaluation metrics
st.subheader("Clustering evaluation")
st.caption(
    "External metrics compare DBSCAN's clusters with the known generated "
    "labels. The silhouette score below excludes noise points."
)

homogeneity = metrics.homogeneity_score(labels_true, labels)
completeness = metrics.completeness_score(labels_true, labels)
v_measure = metrics.v_measure_score(labels_true, labels)
ari = metrics.adjusted_rand_score(labels_true, labels)
ami = metrics.adjusted_mutual_info_score(labels_true, labels)

metric_cols = st.columns(5)
metric_values = [
    ("Homogeneity", homogeneity),
    ("Completeness", completeness),
    ("V-measure", v_measure),
    ("Adjusted Rand Index", ari),
    ("Adjusted Mutual Information", ami),
]
for column, (name, value) in zip(metric_cols, metric_values):
    column.metric(name, f"{value:.3f}")

non_noise_mask = labels != -1
non_noise_labels = labels[non_noise_mask]
if len(np.unique(non_noise_labels)) >= 2:
    silhouette = metrics.silhouette_score(
        X[non_noise_mask], non_noise_labels
    )
    st.metric("Silhouette coefficient (excluding noise)", f"{silhouette:.3f}")
else:
    st.warning(
        "Silhouette coefficient is unavailable: at least two non-noise "
        "clusters are required."
    )

# Data preview and download
st.divider()
st.subheader("Dataset preview")

results_df = pd.DataFrame(
    {
        "Feature 1 (scaled)": X[:, 0],
        "Feature 2 (scaled)": X[:, 1],
        "Original label": labels_true,
        "DBSCAN cluster": labels,
        "Point type": np.where(
            labels == -1,
            "Noise",
            np.where(core_mask, "Core", "Border"),
        ),
    }
)

st.dataframe(results_df.head(20), use_container_width=True)
st.download_button(
    "⬇️ Download clustering results (CSV)",
    data=results_df.to_csv(index=False).encode("utf-8"),
    file_name="dbscan_clustering_results.csv",
    mime="text/csv",
)

with st.expander("How to interpret the parameters"):
    st.markdown(
        """
        - **eps:** maximum distance used to define a point's neighborhood.
        - **min_samples:** minimum number of nearby samples needed for a core point.
        - If `eps` is too small, many points may be marked as noise.
        - If `eps` is too large, separate groups may merge.
        - Tune both values for the dataset; these defaults are only a starting point.
        """
    )

st.caption("Built with Python, Streamlit, NumPy, Matplotlib, Pandas, and Scikit-learn.")
