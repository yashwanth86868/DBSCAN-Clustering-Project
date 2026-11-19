
# DBSCAN Clustering Project

An interactive machine learning project demonstrating **DBSCAN
(Density-Based Spatial Clustering of Applications with Noise)** using
Python, Scikit-learn, and Streamlit.

The project generates synthetic data, scales the features, applies DBSCAN,
evaluates the clustering, and provides an interactive dashboard for exploring
clusters, core points, border points, and noise.

## Project structure

```text
DBSCAN-Clustering-Project/
├── frontend/
│   └── app.py
├── src/
│   └── dbscan_clustering.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Features

- Interactive Streamlit dashboard.
- Adjustable `eps`, `min_samples`, sample count, and cluster spread.
- DBSCAN clustering with Scikit-learn.
- Core, border, and noise point identification.
- Cluster visualization.
- Cluster count and noise-point summary.
- Homogeneity, completeness, V-measure, Adjusted Rand Index (ARI), and
  Adjusted Mutual Information (AMI).
- Silhouette coefficient calculated on non-noise points when at least two
  non-noise clusters exist.
- Download clustering results as a CSV file.

## Requirements

- Python 3.9 or newer
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit

## Run the Streamlit frontend

### 1. Clone your GitHub repository

```bash
git clone https://github.com/YOUR-USERNAME/DBSCAN-Clustering-Project.git
cd DBSCAN-Clustering-Project
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create a virtual environment (recommended)

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Start the frontend

Run this command from the project root folder:

```bash
streamlit run frontend/app.py
```

Streamlit will provide a local address, usually `http://localhost:8501`.
Open that address in your browser if it does not open automatically.

## Run the standalone Python implementation

The original command-line implementation is also included:

```bash
python src/dbscan_clustering.py
```

It prints clustering metrics and saves the cluster plot as
`dbscan_clusters.png`.

## DBSCAN parameters

| Parameter       | Description                                                       |
| --------------- | ----------------------------------------------------------------- |
| `eps`         | Maximum distance between neighboring samples                      |
| `min_samples` | Minimum number of samples in an eps-neighborhood for a core point |

The right values depend on the dataset. Tune the parameters using the
dashboard and inspect both the visualization and evaluation metrics.

## Understanding the point types

- **Core point:** has at least `min_samples` samples in its eps-neighborhood,
  including itself.
- **Border point:** is not a core point but lies within the eps-neighborhood
  of a core point.
- **Noise point:** is not assigned to a cluster and receives label `-1`.

## Evaluation note

The generated dataset includes known original labels, so external metrics
compare DBSCAN's assignments with those labels. The silhouette coefficient
measures separation among the predicted non-noise clusters and excludes
noise points.

## Author

**Yashwanth Balija**

GitHub: https://github.com/yashwanth86868
