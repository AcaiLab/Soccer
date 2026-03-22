import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import matplotlib.pyplot as plt

# ----------------------------
# CONFIG
# ----------------------------
ROOT = Path(".")   
FEATURE_FOLDER_PREFIX = "clip_features"
MAX_SAMPLES_PER_LABEL_PER_MATCH = 200   
RANDOM_STATE = 42

EXCLUDED_LABELS = {
    "start_of_half_game",
    "end_of_half_game",
    "statistics_and_summary",
    "show_added_time"
}

# ----------------------------
# LOAD FRAME EMBEDDINGS
# ----------------------------
X_list = []
y_labels = []
source_info = []

feature_dirs = [
    p for p in ROOT.iterdir()
    if p.is_dir() and p.name.startswith(FEATURE_FOLDER_PREFIX)
]

if not feature_dirs:
    raise RuntimeError(
        f"No feature folders found in {ROOT.resolve()} starting with '{FEATURE_FOLDER_PREFIX}'"
    )

print(f"Found {len(feature_dirs)} feature folders.\n")

for feature_dir in sorted(feature_dirs):
    print(f"Reading from: {feature_dir.name}")

    for npy_file in sorted(feature_dir.glob("*.npy")):
        label = npy_file.stem

        if label in EXCLUDED_LABELS:
            print(f"  Skipping excluded label: {label}")
            continue

        arr = np.load(npy_file)

        # Expected shape: (num_frames, embedding_dim)
        if arr.ndim != 2:
            print(f"  Skipping {npy_file.name} because shape is {arr.shape}, expected 2D")
            continue

        # Optional subsampling for speed and rough class balance
        if len(arr) > MAX_SAMPLES_PER_LABEL_PER_MATCH:
            rng = np.random.default_rng(RANDOM_STATE)
            idx = rng.choice(len(arr), size=MAX_SAMPLES_PER_LABEL_PER_MATCH, replace=False)
            arr = arr[idx]

        X_list.append(arr)
        y_labels.extend([label] * len(arr))
        source_info.extend([f"{feature_dir.name}/{label}"] * len(arr))

        print(f"  Loaded {npy_file.name:30s} shape={arr.shape}")

if not X_list:
    raise RuntimeError("No usable frame embeddings found after filtering.")

X = np.vstack(X_list)
y_labels = np.array(y_labels)
source_info = np.array(source_info)

print("\n" + "=" * 60)
print("DATA SUMMARY")
print("=" * 60)
print(f"Total frame embeddings loaded: {X.shape[0]}")
print(f"Embedding dimension: {X.shape[1]}")
print(f"Number of unique labels: {len(set(y_labels))}")
print("Labels:", sorted(set(y_labels)))

# ----------------------------
# ENCODE LABELS AS INTS
# ----------------------------
unique_labels = sorted(set(y_labels))
label_to_int = {label: i for i, label in enumerate(unique_labels)}
int_to_label = {i: label for label, i in label_to_int.items()}

y_true = np.array([label_to_int[label] for label in y_labels])

# ----------------------------
# K-MEANS
# ----------------------------
n_clusters = len(unique_labels)

print("\nRunning k-means...")
kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=RANDOM_STATE,
    n_init=10
)

cluster_ids = kmeans.fit_predict(X)

# Clustering evaluation
ari = adjusted_rand_score(y_true, cluster_ids)
nmi = normalized_mutual_info_score(y_true, cluster_ids)

print("\n" + "=" * 60)
print("K-MEANS RESULTS")
print("=" * 60)
print(f"Adjusted Rand Index (ARI): {ari:.4f}")
print(f"Normalized Mutual Info (NMI): {nmi:.4f}")

# ----------------------------
# CLUSTER COMPOSITION
# ----------------------------
print("\nCluster compositions:")
for cluster_id in range(n_clusters):
    idx = np.where(cluster_ids == cluster_id)[0]
    labels_in_cluster = y_labels[idx]

    print(f"\nCluster {cluster_id} | size = {len(idx)}")
    if len(idx) == 0:
        continue

    values, counts = np.unique(labels_in_cluster, return_counts=True)
    order = np.argsort(-counts)

    for v, c in zip(values[order], counts[order]):
        pct = 100 * c / len(idx)
        print(f"  {v:25s} : {c:4d} ({pct:5.1f}%)")

# ----------------------------
# PCA BEFORE t-SNE
# ----------------------------
print("\nRunning PCA -> t-SNE...")
pca_dims = min(50, X.shape[1])
pca = PCA(n_components=pca_dims, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X)

tsne = TSNE(
    n_components=2,
    random_state=RANDOM_STATE,
    perplexity=30,
    init="pca",
    learning_rate="auto"
)

X_tsne = tsne.fit_transform(X_pca)

# ----------------------------
# PLOT: TRUE LABELS
# ----------------------------
plt.figure(figsize=(12, 8))

for label in unique_labels:
    idx = y_labels == label
    plt.scatter(
        X_tsne[idx, 0],
        X_tsne[idx, 1],
        s=12,
        alpha=0.7,
        label=label
    )

plt.title("t-SNE of Frame-Level CLIP Embeddings (True Labels)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(fontsize=8, markerscale=1.5, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("tsne_true_labels.png", dpi=300)
plt.show()

# ----------------------------
# PLOT: K-MEANS CLUSTERS
# ----------------------------
plt.figure(figsize=(12, 8))

for cluster_id in range(n_clusters):
    idx = cluster_ids == cluster_id
    plt.scatter(
        X_tsne[idx, 0],
        X_tsne[idx, 1],
        s=12,
        alpha=0.7,
        label=f"Cluster {cluster_id}"
    )

plt.title("t-SNE of Frame-Level CLIP Embeddings (K-Means Clusters)")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend(fontsize=8, markerscale=1.5, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("tsne_kmeans_clusters.png", dpi=300)
plt.show()

print("\nSaved plots:")
print("  tsne_true_labels.png")
print("  tsne_kmeans_clusters.png")