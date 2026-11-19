"""
DBSCAN Clustering Project
-------------------------
Generate synthetic data, scale it, cluster it with DBSCAN, evaluate the
clustering, and visualize core, border, and noise points.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler


def main():
    # 1. Generate sample data
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(
        n_samples=750,
        centers=centers,
        cluster_std=0.4,
        random_state=0,
    )

    # 2. Scale the features
    X = StandardScaler().fit_transform(X)

    # 3. Fit DBSCAN
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)

    # True for core points; False for border points and noise
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True

    labels = db.labels_

    # 4. Count clusters and noise points
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))

    print(f"Estimated number of clusters: {n_clusters}")
    print(f"Estimated number of noise points: {n_noise}")

    # 5. Evaluate clustering against the generated ground-truth labels
    print(f"Homogeneity: {metrics.homogeneity_score(labels_true, labels):.3f}")
    print(f"Completeness: {metrics.completeness_score(labels_true, labels):.3f}")
    print(f"V-measure: {metrics.v_measure_score(labels_true, labels):.3f}")
    print(
        "Adjusted Rand Index: "
        f"{metrics.adjusted_rand_score(labels_true, labels):.3f}"
    )
    print(
        "Adjusted Mutual Information: "
        f"{metrics.adjusted_mutual_info_score(labels_true, labels):.3f}"
    )

    # Silhouette score should be calculated on non-noise points only.
    non_noise_mask = labels != -1
    non_noise_labels = labels[non_noise_mask]

    if len(np.unique(non_noise_labels)) >= 2:
        silhouette = metrics.silhouette_score(
            X[non_noise_mask], non_noise_labels
        )
        print(f"Silhouette Coefficient (excluding noise): {silhouette:.3f}")
    else:
        print("Silhouette Coefficient: unavailable (fewer than 2 non-noise clusters)")

    # 6. Plot clusters: core points are larger; border points are smaller;
    # noise points are black.
    colors = [
        plt.cm.Spectral(each)
        for each in np.linspace(0, 1, max(len(unique_labels), 1))
    ]

    for cluster_label, color in zip(unique_labels, colors):
        if cluster_label == -1:
            color = [0, 0, 0, 1]  # Black for noise

        class_member_mask = labels == cluster_label

        # Core points
        core_points = X[class_member_mask & core_samples_mask]
        plt.plot(
            core_points[:, 0],
            core_points[:, 1],
            "o",
            markerfacecolor=tuple(color),
            markeredgecolor="k",
            markersize=14,
        )

        # Border points and noise points
        non_core_points = X[class_member_mask & ~core_samples_mask]
        plt.plot(
            non_core_points[:, 0],
            non_core_points[:, 1],
            "o",
            markerfacecolor=tuple(color),
            markeredgecolor="k",
            markersize=6,
        )

    plt.title(f"DBSCAN: estimated number of clusters = {n_clusters}")
    plt.xlabel("Scaled feature 1")
    plt.ylabel("Scaled feature 2")
    plt.tight_layout()
    plt.savefig("dbscan_clusters.png", dpi=300)
    print("Plot saved as dbscan_clusters.png")
    plt.show()


if __name__ == "__main__":
    main()
