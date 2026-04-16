# ========== Third-Party Libraries ==========
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE


def visualize_augmented_tsne(original_data, new_samples_list, sample_size=2000):
    """
    Visualizes the original vs. augmented samples using t-SNE.
    """
    print("📊 Executing t-SNE visualization for augmented samples...")

    if isinstance(original_data, pd.DataFrame):
        original_data = original_data.values

    all_new = np.vstack(new_samples_list)
    all_data = np.vstack((original_data, all_new))

    features = all_data[:, :-1]
    labels = all_data[:, -1].astype(int)

    num_original = len(original_data)

    # Handle random downsampling for massive datasets to speed up t-SNE
    if len(features) > sample_size:
        indices = np.random.choice(len(features), size=sample_size, replace=False)
        features = features[indices]
        labels = labels[indices]
        # Create a boolean mask: True if the sampled index belongs to the original dataset
        is_original = indices < num_original
    else:
        is_original = np.arange(len(features)) < num_original

    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embedding = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))

    # 1. Plot original samples (Circles)
    scatter = plt.scatter(
        embedding[is_original, 0], embedding[is_original, 1],
        c=labels[is_original], cmap="tab10",
        marker='o', s=20, alpha=0.6,
        edgecolors='k', label="Original"
    )

    # 2. Plot augmented samples (Crosses)
    is_augmented = ~is_original
    if np.any(is_augmented):
        plt.scatter(
            embedding[is_augmented, 0], embedding[is_augmented, 1],
            c=labels[is_augmented], cmap="tab10",
            marker='x', s=30, linewidths=1.5, alpha=0.9, label="Augmented"
        )

    plt.title("t-SNE Visualization: Original vs Augmented Samples", fontsize=14, fontweight='bold')

    # Extract handles for classes, avoiding duplicating the markers in the legend
    handles, legend_labels = scatter.legend_elements()
    plt.legend(handles, legend_labels, title="Class", loc="best")

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()


def visualize_embedding_comparison(original_X, mapped_X, labels, title):
    """
    Compares the original feature space vs. the manifold-mapped feature space.
    """
    # Downsample for visualization speed and clarity
    n = min(1000, len(original_X))
    idx = np.random.choice(len(original_X), size=n, replace=False)

    X_ori = original_X[idx]
    X_map = mapped_X[idx]
    y = labels[idx]

    tsne_ori = TSNE(n_components=2, random_state=42).fit_transform(X_ori)
    tsne_map = TSNE(n_components=2, random_state=42).fit_transform(X_map)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    titles = ["Original Space", "Embedded Space"]
    embeddings = [tsne_ori, tsne_map]

    scatter = None
    for ax, emb, title_part in zip(axes, embeddings, titles):
        scatter = ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap="tab10", s=20, alpha=0.6)
        ax.set_title(f"{title} - {title_part}", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)

    if scatter is not None:
        plt.legend(*scatter.legend_elements(), title="Class", loc='best')

    plt.tight_layout()
    plt.show()