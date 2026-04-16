# ========== Standard Library ==========
import os
import time
import warnings
from collections import Counter, defaultdict
from statistics import mean, stdev

# ========== Third-Party Libraries ==========
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import ADASYN
from lightgbm import LGBMClassifier
from scipy.spatial import KDTree as SciPyKDTree
from scipy.spatial.distance import pdist
from scipy.sparse import SparseEfficiencyWarning
from scipy.stats import ttest_rel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.manifold import LocallyLinearEmbedding, Isomap
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    cohen_kappa_score, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KDTree, NearestNeighbors
from sklearn.svm import SVC
from xgboost import XGBClassifier

# ========== PyTorch ==========
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import is_available
from torch.utils.data import DataLoader, TensorDataset

# ========== Custom Modules ==========
from fig import visualize_embedding_comparison, visualize_augmented_tsne
from model2 import train_mlp_once, measures_of_classify
from manifold_mapperLSTMMLPXGB import neighborhood_Measure_mm

# ========== Warnings ==========
warnings.simplefilter("ignore", SparseEfficiencyWarning)


# ==========================================
# 1. New Classifier Wrappers (XGBoost, LightGBM)
# ==========================================
def train_xgb_once(X_train, y_train, X_test, y_test, num_classes, device_id=None):
    """XGBoost Classifier Wrapper"""
    # Convert to pure numpy arrays to thoroughly eliminate feature names warnings
    X_tr_clean = np.asarray(X_train)
    X_te_clean = np.asarray(X_test)

    # Random seed is controlled by the global numpy seed to ensure slight variance across runs
    seed = np.random.randint(0, 10000)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softmax',
        num_class=num_classes,
        eval_metric='mlogloss',
        random_state=seed,
        n_jobs=-1
    )
    model.fit(X_tr_clean, y_train)
    y_pred = model.predict(X_te_clean)
    return y_test, y_pred


def train_lgbm_once(X_train, y_train, X_test, y_test, num_classes, device_id=None):
    """LightGBM Classifier Wrapper"""
    # Convert to pure numpy arrays to thoroughly eliminate feature names warnings
    X_tr_clean = np.asarray(X_train)
    X_te_clean = np.asarray(X_test)

    seed = np.random.randint(0, 10000)

    model = LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='multiclass',
        num_class=num_classes,
        random_state=seed,
        n_jobs=-1,
        verbose=-1
    )
    model.fit(X_tr_clean, y_train)
    y_pred = model.predict(X_te_clean)
    return y_test, y_pred


# ==========================================
# 2. Deep Learning Sequence Model (LSTM) Wrapper
# ==========================================
class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def train_lstm_once(X_train, y_train, X_test, y_test, num_classes, device_id=None, epochs=20, batch_size=256):
    """LSTM Classifier Wrapper (only for downstream validation)"""
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() and device_id is not None else "cpu")

    X_tr_clean = np.asarray(X_train)
    X_te_clean = np.asarray(X_test)

    X_tr_tensor = torch.tensor(X_tr_clean, dtype=torch.float32).unsqueeze(1).to(device)
    y_tr_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_te_tensor = torch.tensor(X_te_clean, dtype=torch.float32).unsqueeze(1).to(device)

    dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleLSTM(input_dim=X_train.shape[1], hidden_dim=64, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        test_outputs = model(X_te_tensor)
        _, y_pred = torch.max(test_outputs.data, 1)

    return y_test, y_pred.cpu().numpy()


# ==========================================
# 3. Statistical Testing and Automated Plotting Functions
# ==========================================
def perform_stat_test_and_plot(results_dict, metric_name="Macro F1"):
    plot_data = []
    summary_data = []  # Used to export CSV table

    print(f"\n📈 ===== Statistical Significance Test ({metric_name}) (Paired T-test) =====")
    for model_name, (base_list, aug_list) in results_dict.items():
        # Paired T-test
        t_stat, p_val = ttest_rel(base_list, aug_list)
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"

        print(f"[{model_name}] Baseline vs MACO -> p-value: {p_val:.4e} ({significance})")

        # Format data for plotting
        for val in base_list:
            plot_data.append({"Model": model_name, "Method": "Baseline", metric_name: val})
        for val in aug_list:
            plot_data.append({"Model": model_name, "Method": "MACO Augmented", metric_name: val})

        # Format data for tabulation
        base_mean, base_std = mean(base_list), stdev(base_list)
        aug_mean, aug_std = mean(aug_list), stdev(aug_list)
        improvement = (aug_mean - base_mean) / base_mean * 100

        summary_data.append({
            "Classifier": model_name,
            "Baseline (Mean ± Std)": f"{base_mean:.4f} ± {base_std:.4f}",
            "MACO (Mean ± Std)": f"{aug_mean:.4f} ± {aug_std:.4f}",
            "Improvement (%)": f"+{improvement:.2f}%",
            "p-value": f"{p_val:.4e}",
            "Significance": significance
        })

    df_plot = pd.DataFrame(plot_data)
    df_summary = pd.DataFrame(summary_data)

    # 💾 Export 1: Statistical Table (CSV)
    table_filename = "Statistical_Significance_Report.csv"
    df_summary.to_csv(table_filename, index=False)
    print(f"✅ Statistical table saved to: {os.path.abspath(table_filename)}")

    # 🎨 Export 2: Bar Plot with Error Bars
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    ax1 = sns.barplot(x="Model", y=metric_name, hue="Method", data=df_plot, capsize=.1, palette="Set2")
    plt.title(f"Generalization Across Classifiers: {metric_name}", fontsize=14, fontweight='bold')
    plt.ylabel(metric_name, fontsize=12)
    plt.xlabel("Classifier Architecture", fontsize=12)
    plt.legend(title="Data Setup", loc='lower right')
    bar_filename = f"Generalization_BarPlot_{metric_name.replace(' ', '_')}.png"
    plt.savefig(bar_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Bar plot saved to: {os.path.abspath(bar_filename)}")

    # 🎨 Export 3: Box Plot + Scatter Plot (specifically to show randomness and variance)
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="ticks")
    ax2 = sns.boxplot(x="Model", y=metric_name, hue="Method", data=df_plot, palette="pastel", width=0.6, fliersize=0)

    # Add scatter plot to show specific distribution of the 5 runs
    sns.stripplot(x="Model", y=metric_name, hue="Method", data=df_plot, dodge=True, color='black', alpha=0.6, ax=ax2)

    # Remove duplicate legend
    handles, labels = ax2.get_legend_handles_labels()
    plt.legend(handles[0:2], labels[0:2], title="Data Setup", loc='lower right')

    plt.title(f"Robustness Across Random Seeds (5 Runs): {metric_name}", fontsize=14, fontweight='bold')
    plt.ylabel(metric_name, fontsize=12)
    plt.xlabel("Classifier Architecture", fontsize=12)
    box_filename = f"Robustness_BoxPlot_{metric_name.replace(' ', '_')}.png"
    plt.savefig(box_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Box plot saved to: {os.path.abspath(box_filename)}")


def estimate_local_sigma(x, global_features, global_labels, y, k=5):
    indices = np.where(global_labels == y)[0]
    if len(indices) < 2:
        return 1.0
    tree = KDTree(global_features[indices])
    dists, _ = tree.query(x.reshape(1, -1), k=min(k + 1, len(indices)))
    avg_dist = np.mean(dists[0][1:])  # Exclude self
    return max(avg_dist, 1e-6)


def get_top_opp_classes(global_features, global_labels, target_class, top_n=2):
    cls_centers = {}
    target_mask = (global_labels == target_class)
    if np.sum(target_mask) == 0:
        return []

    target_center = np.mean(global_features[target_mask], axis=0)

    for cls in np.unique(global_labels):
        if cls == target_class:
            continue
        cls_mask = (global_labels == cls)
        if np.sum(cls_mask) == 0:
            continue
        center = np.mean(global_features[cls_mask], axis=0)
        cls_centers[cls] = np.linalg.norm(center - target_center)

    sorted_cls = sorted(cls_centers.items(), key=lambda x: x[1])
    return [cls for cls, _ in sorted_cls[:top_n]]


def cen_mar_func_top_opp(data, global_features, global_labels,
                         k_sim=3, k_opp=3, top_n_opp_classes=2, verbose=True):
    def stable_gaussian_weight(dist, sigma):
        sigma = max(sigma, 1e-2)
        weight = np.exp(-dist ** 2 / (2 * sigma ** 2))
        weight[weight < 1e-4] = 0
        return weight

    features = data[:, :-1]
    labels = data[:, -1]
    r = features.shape[0]
    Degree = np.zeros((r, 2))  # Centrality, Marginality

    for i in range(r):
        x = features[i]
        y = labels[i]

        sigma = estimate_local_sigma(x, global_features, global_labels, y)

        # Intra-class neighbors
        sim_indices_all = np.where(global_labels == y)[0]
        if len(sim_indices_all) >= 1:
            sim_tree = KDTree(global_features[sim_indices_all])
            _, sim_idx_local = sim_tree.query(x.reshape(1, -1), k=min(k_sim, len(sim_indices_all)))
            sim_indices = sim_indices_all[sim_idx_local[0]]
        else:
            sim_indices = []

        # Inter-class neighbors
        top_opp_classes = get_top_opp_classes(global_features, global_labels, y, top_n=top_n_opp_classes)
        opp_indices_all = np.concatenate(
            [np.where(global_labels == c)[0] for c in top_opp_classes if np.sum(global_labels == c) > 0])
        if len(opp_indices_all) >= 1:
            opp_tree = KDTree(global_features[opp_indices_all])
            _, opp_idx_local = opp_tree.query(x.reshape(1, -1), k=min(k_opp, len(opp_indices_all)))
            opp_indices = opp_indices_all[opp_idx_local[0]]
        else:
            opp_indices = []

        if verbose:
            print(
                f"🔍 Sample {i} Class {y}: Intra-class neighbors={len(sim_indices)}, Inter-class neighbors={len(opp_indices)}")

        if len(sim_indices) == 0 and len(opp_indices) == 0:
            if verbose:
                print(f"⚠️ Sample {i} has no valid neighbors, Degree defaults to 0")
            continue

        all_indices = np.concatenate([sim_indices, opp_indices])
        all_labels = global_labels[all_indices]
        all_features = global_features[all_indices]
        all_distances = np.linalg.norm(all_features - x, axis=1)
        weights = stable_gaussian_weight(all_distances, sigma)

        sim_mask = (all_labels == y)
        opp_mask = (all_labels != y)

        Degree[i, 0] = np.sum(weights[sim_mask]) if np.any(sim_mask) else 0.0
        Degree[i, 1] = np.sum(weights[opp_mask]) if np.any(opp_mask) else 0.0

        if verbose and i < 1:
            print(f"🎯 σ={sigma:.4f}, Distance={np.round(all_distances, 2)}")
            print(f"🎯 Weights: {np.round(weights, 4)}")
            print(f"🎯 center sum: {Degree[i, 0]:.4f}, margin sum: {Degree[i, 1]:.4f}")

        if verbose and i < 1:
            print(f"🔍 Sample {i} Class {y}:")
            print(f"    Candidate intra-class samples: {len(sim_indices_all)}, Selected: {len(sim_indices)}")
            print(
                f"    Candidate inter-classes: {top_opp_classes}, Sample count: {len(opp_indices_all) if 'opp_indices_all' in locals() else 0}, Selected: {len(opp_indices)}")

        if np.all(weights == 0) and verbose:
            print(
                f"⚠️ Sample {i} has all neighbor weights at 0, possibly σ={sigma:.4f} is too small or neighbors are too far")

    return Degree


def select_diverse_samples(features, labels, weights, filter_ratio=0.8, n_clusters=3, min_samples_no_cluster=50):
    selected_indices = []
    classes = np.unique(labels)

    for cls in classes:
        cls_mask = (labels == cls)
        cls_features = features[cls_mask]
        cls_weights = weights[cls_mask]
        cls_indices = np.where(cls_mask)[0]
        cls_size = len(cls_features)

        # Minority class, retain all samples directly
        if cls_size < min_samples_no_cluster:
            print(f"📌 Class {cls} has few samples ({cls_size}), skipping clustering and filtering, retaining all")
            selected_indices.extend(cls_indices)
            continue

        # Normal class undergoes filtering and clustering process
        if cls_size < n_clusters:
            top_n = max(1, int(cls_size * filter_ratio))
            top_idx = np.argsort(-cls_weights)[:top_n]
            selected_indices.extend(cls_indices[top_idx])
            continue

        # Clustering + per-cluster filtering
        kmeans = KMeans(n_clusters=n_clusters, n_init=10)
        cluster_labels = kmeans.fit_predict(cls_features)

        for cl in range(n_clusters):
            cluster_mask = (cluster_labels == cl)
            if not np.any(cluster_mask):
                continue
            cluster_weights = cls_weights[cluster_mask]
            cluster_idx_in_cls = np.where(cluster_mask)[0]
            top_n = max(1, int(len(cluster_weights) * filter_ratio))
            top_cluster_idx = np.argsort(-cluster_weights)[:top_n]
            selected_indices.extend(cls_indices[cluster_idx_in_cls[top_cluster_idx]])

    return np.array(selected_indices)


def over_multi_manifold(data, global_data, mapper=None, min_class_size=5,
                        use_score_filter=True, filter_ratio=0.8):
    labels = data[:, -1]
    classes = np.unique(labels)

    class_counts = {cls: np.sum(labels == cls) for cls in classes}
    if any(v < min_class_size for v in class_counts.values()):
        for cls, cnt in class_counts.items():
            if cnt < min_class_size:
                print(f"⚠️ Class {cls} has only {cnt} samples, skipping multi-manifold modeling for this class.")
        return np.array([]), np.array([]), np.array([])

    manifold, all_data_map, mapper = neighborhood_Measure_mm(data, mapper=mapper)
    num_mappings = len(all_data_map)
    r = data.shape[0]

    global_mapped_all = mapper.transform(global_data[:, :-1])
    Degree = np.zeros((r, 2, num_mappings))

    for j in range(num_mappings):
        mapping_type = manifold[0]['type'][j]
        mappedX = all_data_map[j]['all_x']

        map_data = np.hstack((mappedX, data[:, -1].reshape(-1, 1)))
        global_features_mapped = global_mapped_all[mapping_type]

        degree_result = cen_mar_func_top_opp(
            map_data,
            global_features_mapped,
            global_data[:, -1],
            k_sim=3,  # Baseline 3
            k_opp=3,  # Baseline 3
            top_n_opp_classes=3,  # Baseline 3
            verbose=False
        )

        if np.all(degree_result[:, 0] == 0) and np.all(degree_result[:, 1] == 0):
            print(
                f"⚠️ Degree all 0 - mapping method {mapping_type} invalid for all samples, or abnormal neighbor structure")

        Degree[:, 0, j] = degree_result[:, 0]
        Degree[:, 1, j] = degree_result[:, 1]

    weighted_cen = np.zeros((r, 1))
    weighted_mar = np.zeros((r, 1))

    # Ensure class order matches manifold
    label_to_manifold_index = {cls: idx for idx, cls in enumerate(classes)}
    for i in range(r):
        cls_label = data[i, -1]
        if cls_label not in label_to_manifold_index:
            continue
        cls_index = label_to_manifold_index[cls_label]

        alpha = manifold[cls_index]['alpha']
        if len(alpha) != num_mappings:
            print(f"⚠️ Length of α does not match Degree mappings, skipping sample {i}")
            continue

        weighted_cen[i, 0] = np.dot(alpha, Degree[i, 0, :])
        weighted_mar[i, 0] = np.dot(alpha, Degree[i, 1, :])

    weight = Degree[:, 1] / (Degree[:, 0] + 1e-6)  # W = mar / (cen + ε)

    if use_score_filter:
        # Use all data instead of only subset where weight > 0
        weight_marginal = weight
        data_marginal = data
        label_marginal = data[:, -1]

        if len(weight_marginal) == 0:
            print("⚠️ No marginal samples available for filtering, returning empty")
            return np.array([]), np.array([]), np.array([])

        selected_indices = select_diverse_samples(
            data_marginal[:, :-1],
            label_marginal,
            weight_marginal[:, 0],
            filter_ratio=filter_ratio,
            n_clusters=3,
            min_samples_no_cluster=100
        )

        sorted_data = data_marginal[selected_indices]
        sorted_weight = weight_marginal[selected_indices]
        sort_label = label_marginal[selected_indices]
        print(f"🎯 Using score filter: top {filter_ratio * 100:.0f}%, retaining {len(sorted_weight)} samples")

    else:
        # Default logic: use all samples (no filtering)
        sorted_indices = np.argsort(-weight[:, 0])
        sorted_weight = weight[sorted_indices]
        sorted_data = data[sorted_indices, :]
        sort_label = data[sorted_indices, -1]

    return sorted_weight, sorted_data, sort_label


def average_distance(minority_data):
    if minority_data.shape[0] < 2:
        return 0.0
    distances = pdist(minority_data, metric='euclidean')
    return np.mean(distances)


def gradual_overSampling_func_multi_plus(
        sorted_weight, sorted_data, sort_label, target_class,
        alpha=1.0, beta=0.1, k=5, max_iter=20, eps=0.005,
        n_generate_per_sample=1, smote_ratio=0.5, enable_smote=False
):
    r, c = sorted_data.shape
    class_data = sorted_data[sorted_data[:, c - 1] == target_class, :]
    if class_data.shape[0] == 0:
        print(f"⚠️ Class {target_class} has no marginal samples, skipping augmentation.")
        return np.array([]), 1.0  # fallback_ratio=1 indicates failure

    print(f"\n✅ [Class {target_class}] Marginal samples used for oversampling: {class_data.shape[0]}")

    features = np.array(class_data[:, :-1], dtype=np.float64)
    if features.ndim != 2 or features.shape[1] == 0:
        print(f"❌ Feature extraction failed: invalid dimensions {features.shape}")
        return np.array([]), 1.0

    nc, feature_dim = features.shape
    if nc <= 1:
        print(f"⚠️ Class {target_class} has too few samples to establish neighbor structure.")
        return np.array([]), 1.0

    k_eff = min(k, nc - 1)
    if k_eff < 1:
        print(f"⚠️ Class {target_class} has insufficient available samples to establish neighbor structure.")
        return np.array([]), 1.0

    nbrs = NearestNeighbors(n_neighbors=k_eff + 1).fit(features)
    _, indices = nbrs.kneighbors(features)
    Idx = indices[:, 1:]

    ave_dist = average_distance(features)
    w = max(ave_dist * 0.05, 1e-3)

    generated_samples = []
    converge_iters = []
    fallback_count = 0
    smote_count = 0
    dict_success_count = 0
    dict_fallback_count = 0

    for i in range(nc):
        U = features[Idx[i, :], :].T
        center = features[i, :]

        for _ in range(n_generate_per_sample):
            if enable_smote and (np.random.rand() < smote_ratio):
                neighbor_idx = np.random.choice(Idx[i])
                neighbor_vec = features[neighbor_idx]
                lam = np.random.rand()
                x_i = lam * center + (1 - lam) * neighbor_vec
                generated_samples.append(x_i)
                converge_iters.append(0)
                smote_count += 1
                continue

            v_i = np.random.dirichlet(np.ones(k_eff) * 0.5)
            retry_count = 0
            max_retry = 5
            success = False

            for iter_id in range(max_iter):
                x_i = np.dot(U, v_i)

                if any(np.linalg.norm(x_i - features[Idx[i, j], :]) < w for j in range(k_eff)):
                    retry_count += 1
                    if retry_count >= max_retry:
                        generated_samples.append(x_i)
                        converge_iters.append(iter_id)
                        dict_success_count += 1
                        success = True
                        break
                    continue

                try:
                    v_candidate, *_ = np.linalg.lstsq(U, x_i, rcond=None)
                    v_candidate = np.maximum(v_candidate, 0)
                    if np.sum(v_candidate) == 0:
                        v_i = np.random.dirichlet(np.ones(k_eff) * 0.5)
                        continue
                    new_v_i = v_candidate / np.sum(v_candidate)
                except np.linalg.LinAlgError:
                    break

                if iter_id > 4 and np.linalg.norm(new_v_i - v_i) <= eps:
                    generated_samples.append(x_i)
                    converge_iters.append(iter_id)
                    dict_success_count += 1
                    success = True
                    break

                if iter_id == max_iter - 1:
                    generated_samples.append(x_i)
                    converge_iters.append(iter_id)
                    dict_success_count += 1
                    success = True
                    break

                v_i = new_v_i

            if not success:
                generated_samples.append(np.dot(U, v_i))
                converge_iters.append(max_iter)
                fallback_count += 1
                dict_fallback_count += 1

    if len(generated_samples) == 0:
        print(f"⚠️ Class {target_class} generated no augmented samples.")
        return np.array([]), 1.0

    Over = np.array(generated_samples)
    label_newdata = np.full((Over.shape[0], 1), target_class, dtype=sorted_data.dtype)
    Over_Sampling_Data = np.hstack((Over, label_newdata))
    Over_Sampling_Data = np.unique(Over_Sampling_Data, axis=0)

    if converge_iters:
        converge_iters = np.array(converge_iters)
        print(f"📊 Total interpolations: {len(converge_iters)}")
        print(
            f"📊 Average interpolation iterations: {np.mean(converge_iters):.2f}, Max iterations: {np.max(converge_iters)}")
        print(f"📌 Fallback (unconverged) interpolated samples: {fallback_count}")
        print(f"📊 Interpolation method statistics:")
        print(f"   ├─ SMOTE interpolations            : {smote_count}")
        print(f"   ├─ Dictionary interpolation successes: {dict_success_count}")
        print(f"   └─ Dictionary interpolation fallbacks: {dict_fallback_count}")

    return Over_Sampling_Data


def over_classify_mlp(data, train_mlp_once_func,
                      min_class_size=5, max_rounds=3, kf=2, iter_num=2,
                      save_filename="augmented_data.csv"):
    if isinstance(data, pd.DataFrame):
        data = data.values

    labels = data[:, -1].astype(int)
    all_precision, all_recall, all_f1, all_acc = [], [], [], []
    final_new_samples = []

    # 🌟 Ultimate fix: Use global seed-based random numbers to ensure cross-validation splits in 5-run experiments are truly independent
    skf = StratifiedKFold(n_splits=kf, shuffle=True, random_state=np.random.randint(0, 10000))
    split_indices = list(skf.split(data[:, :-1], labels))

    for j in range(iter_num):
        print(f"\n🔁 Iteration {j + 1}/{iter_num}...")
        for i, (train_idx, test_idx) in enumerate(split_indices):
            print(f"   ▶️ Fold {i + 1}/{kf} Cross-Validation...")
            d, t = data[train_idx], data[test_idx]

            label_counts = Counter(d[:, -1])
            counts = np.array(list(label_counts.values()))
            mean_count = np.mean(counts)
            threshold = 0.6 * mean_count
            valid_classes = [cls for cls, count in label_counts.items() if count < threshold]

            print(f"      - Current class sample counts: {dict(label_counts)}")
            print(f"      - Average sample count: {mean_count:.0f}, Minority class threshold: {threshold:.0f}")
            print(f"      - Dynamically selected classes for augmentation: {valid_classes}")

            if not valid_classes:
                print("⚠️ No valid minority classes, skipping augmentation")
                continue

            X_d, y_d = d[:, :-1], d[:, -1].astype(int)
            X_t, y_t = t[:, :-1], t[:, -1].astype(int)
            num_classes = len(np.unique(np.concatenate((y_d, y_t))))
            y_true, y_pred = train_mlp_once_func(X_d, y_d, X_t, y_t, num_classes, device_id=0)

            cm_before = confusion_matrix(y_true, y_pred, labels=range(num_classes))
            r0, p0, f0, _, acc0 = measures_of_classify(cm_before)
            best_f1, best_acc = f0, acc0
            over_data = d.copy()

            cumulative_new_samples = []  # Record all augmented rounds' samples

            print(f"      📊 Original F1: {f0:.4f} | Accuracy: {acc0:.4f}")

            for round_num in range(1, max_rounds + 1):
                print(f"      🔁 Augmentation round {round_num}...")
                new_data_all_classes = []

                for cls in valid_classes:
                    cls_data = over_data[over_data[:, -1] == cls]
                    if cls_data.shape[0] < min_class_size:
                        print(f"        ⚠️ Class {cls} has too few samples, skipping augmentation")
                        continue

                    sorted_weight, sorted_data, sort_label = over_multi_manifold(cls_data, d)

                    if sorted_data.ndim != 2 or sorted_data.shape[0] == 0:
                        print(f"        ⚠️ Class {cls} cannot execute over_multi_manifold, skipping augmentation")
                        continue

                    new_cls_data = gradual_overSampling_func_multi_plus(
                        sorted_weight, sorted_data, sort_label,
                        target_class=cls,
                        n_generate_per_sample=2,
                        enable_smote=True,
                        smote_ratio=0.8
                    )

                    if new_cls_data.size > 0:
                        target_count = int(np.mean(list(label_counts.values())))
                        current_count = label_counts[cls]
                        needed = target_count - current_count
                        if needed <= 0:
                            continue
                        max_add = min(needed, new_cls_data.shape[0])
                        new_cls_data = new_cls_data[:max_add]

                        print(
                            f"        ✅ Class {cls} augmented samples: {new_cls_data.shape[0]} (Original: {label_counts[cls]})")
                        new_data_all_classes.append(new_cls_data)
                    else:
                        print(f"        ⚠️ Class {cls} generated no augmented samples")

                if not new_data_all_classes:
                    print("        ❌ No augmented samples generated in this round, terminating early")
                    break

                new_samples = np.vstack(new_data_all_classes)
                temp_over_data = np.vstack((over_data, new_samples))
                temp_over_data = np.unique(temp_over_data, axis=0)

                X_temp, y_temp = temp_over_data[:, :-1], temp_over_data[:, -1].astype(int)
                y_true_temp, y_pred_temp = train_mlp_once_func(X_temp, y_temp, X_t, y_t, num_classes, device_id=0)
                cm_after = confusion_matrix(y_true_temp, y_pred_temp, labels=range(num_classes))
                r1, p1, f1, _, acc1 = measures_of_classify(cm_after)

                print(f"        🧪 F1 after augmentation: {f1:.4f} | Accuracy: {acc1:.4f}")

                if f1 >= best_f1:
                    best_f1, best_acc = f1, acc1
                    over_data = temp_over_data
                    cumulative_new_samples.append(new_samples)
                else:
                    print("        ⚠️ No significant improvement from augmentation, stopping iteration")
                    break

            # Fold augmentation complete, merge augmented samples from all rounds
            if cumulative_new_samples:
                all_new_samples = np.vstack(cumulative_new_samples)
                final_new_samples.append(all_new_samples)
                print(f"   📌 Cumulative total of augmented samples for this fold: {all_new_samples.shape[0]}")
            else:
                print("   ⚠️ No augmented samples saved for this fold")

            all_precision.append(p1)
            all_recall.append(r1)
            all_f1.append(best_f1)
            all_acc.append(best_acc)
            print(f"   ✅ Best F1 in fold: {best_f1:.4f} | Accuracy: {best_acc:.4f}")

    print("\n✅ All cross-validations complete")
    print(f"📌 Average F1: {mean(all_f1):.4f}, Average Accuracy: {mean(all_acc):.4f}")

    # Save augmented samples (using dynamically passed save_filename)
    if final_new_samples:
        final_new_samples_filtered = [s for s in final_new_samples if s.size > 0]
        if final_new_samples_filtered:
            head_df, total_count = save_final_augmented_data(data, final_new_samples_filtered, filename=save_filename)
            print(f"📦 Total final training samples: {total_count}")

    return mean(all_precision), mean(all_recall), mean(all_f1), float('nan'), mean(all_acc)


def save_final_augmented_data(original_data, final_new_samples, filename):
    print(f"📦 Saving final augmented training set to: {filename} ...")
    print(f"   - Original sample count: {original_data.shape[0]}")

    all_new_samples = np.vstack(final_new_samples)
    print(f"   - Augmented sample count (before merge): {sum([s.shape[0] for s in final_new_samples])}")

    # Merge original and augmented data
    final_data = np.vstack((original_data, all_new_samples))
    print(f"   - Total sample count after merge: {final_data.shape[0]}")

    # Use original column names (assuming original is DataFrame format)
    if isinstance(original_data, pd.DataFrame):
        col_names = list(original_data.columns)
    else:
        num_features = final_data.shape[1] - 1
        col_names = [f"feature_{i + 1}" for i in range(num_features)] + ["label"]

    df_save = pd.DataFrame(final_data, columns=col_names)
    df_save.to_csv(filename, index=False)
    print(f"✅ Augmented training set successfully saved as: {filename}\n")

    return df_save.head(), final_data.shape[0]


def compute_gmeans(y_true, y_pred):
    """
    Calculate G-means = sqrt(Macro Precision × Macro Recall)
    Suitable for overall estimation of multi-class tasks
    """
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    gmeans = np.sqrt(macro_precision * macro_recall)
    return gmeans


def conditional_stratified_sample(df, label_col, frac, min_samples=100, random_state=None):
    """Conditional sampling by class; if class samples are fewer than min_samples, retain all"""
    sampled = []
    for label, group in df.groupby(label_col):
        if len(group) < min_samples:
            sampled.append(group)
        else:
            sampled.append(group.sample(frac=frac, random_state=random_state))
    return pd.concat(sampled).sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def main(n_runs=5):
    print(
        f"\n📊 Executing {n_runs} full cross-classifier experiments, will automatically generate plots and statistical reports...\n")

    classifiers = {
        "MLP": (train_mlp_once, True),
        "XGBoost": (train_xgb_once, True),
        "LightGBM": (train_lgbm_once, True),
        "LSTM": (train_lstm_once, False)
    }

    f1_results_for_plot = {}
    sample_tracking_data = []  # New: Used to track generated sample counts

    train = pd.read_csv("multi_train.csv", index_col=0)
    test = pd.read_csv("multi_test.csv", index_col=0)

    feature_cols = train.columns[:-1]
    label_col = train.columns[-1]

    X_train = train[feature_cols].values
    y_train = train[label_col].values.astype(int)
    X_test = test[feature_cols].values
    y_test = test[label_col].values.astype(int)

    num_classes = len(np.unique(np.concatenate([y_train, y_test])))
    train_data_raw = np.hstack((X_train, y_train.reshape(-1, 1)))
    orig_samples_count = train_data_raw.shape[0]

    best_augmented_data_for_lstm = None

    for model_name, (train_func, run_maco) in classifiers.items():
        print(f"\n{'=' * 50}")
        print(f"🚀 Starting evaluation of classifier architecture: {model_name}")
        print(f"{'=' * 50}")

        base_f1_list = []
        aug_f1_list = []

        for run in range(n_runs):
            # Set random seed to ensure statistical independence and variance across the 5 runs
            # Also ensuring global reproducibility (run=0 is always seed 42)
            np.random.seed(42 + run)

            print(f"\n--- 🔄 Run {run + 1}/{n_runs} ({model_name}) ---")
            current_save_name = f"augmented_{model_name}_run{run}.csv"

            # 1. Baseline Experiment
            _, y_pred_base = train_func(X_train, y_train, X_test, y_test, num_classes)
            base_f1 = f1_score(y_test, y_pred_base, average='macro', zero_division=0)
            base_f1_list.append(base_f1)

            # 2. Augmentation Experiment
            if run_maco:
                print(f"  ▶️ {model_name} Starting MACO multi-round dynamic oversampling...")
                _ = over_classify_mlp(train_data_raw, train_func, save_filename=current_save_name)

                if os.path.exists(current_save_name):
                    aug_data = pd.read_csv(current_save_name).values
                    best_augmented_data_for_lstm = aug_data
                else:
                    print(
                        f"  ⚠️ Generated {current_save_name} not found, augmentation might have failed, using original data.")
                    aug_data = train_data_raw
            else:
                borrowed_file = f"augmented_XGBoost_run{run}.csv"
                print(f"  ▶️ {model_name} Only used for downstream testing, attempting to load: {borrowed_file} ...")
                if os.path.exists(borrowed_file):
                    aug_data = pd.read_csv(borrowed_file).values
                else:
                    print("  ⚠️ Warning: No borrowable augmented data found, falling back to original data.")
                    aug_data = train_data_raw

            X_aug = aug_data[:, :-1]
            y_aug = aug_data[:, -1].astype(int)

            _, y_pred_aug = train_func(X_aug, y_aug, X_test, y_test, num_classes)
            aug_f1 = f1_score(y_test, y_pred_aug, average='macro', zero_division=0)
            aug_f1_list.append(aug_f1)

            # 📈 Track sample generation count
            aug_samples_count = aug_data.shape[0]
            synthetic_added = aug_samples_count - orig_samples_count
            sample_tracking_data.append({
                "Model": model_name,
                "Run": run + 1,
                "Original Dataset Size": orig_samples_count,
                "Augmented Dataset Size": aug_samples_count,
                "Synthetic Samples Added": synthetic_added
            })

            print(f"  📊 {model_name} Run {run + 1} Results -> Baseline F1: {base_f1:.4f} | MACO F1: {aug_f1:.4f}")

        f1_results_for_plot[model_name] = (base_f1_list, aug_f1_list)
        print(f"\n✅ {model_name} Experiment complete.")
        print(f"  Mean Baseline F1: {mean(base_f1_list):.4f} ± {stdev(base_f1_list):.4f}")
        print(f"  Mean MACO F1    : {mean(aug_f1_list):.4f} ± {stdev(aug_f1_list):.4f}")

    # Execute statistical tests and generate all plots
    perform_stat_test_and_plot(f1_results_for_plot, metric_name="Macro F1")

    # 💾 Export 4: Sample Generation Count Report (CSV)
    df_samples = pd.DataFrame(sample_tracking_data)
    # Calculate the average number of samples added per run
    df_samples_summary = df_samples.groupby("Model").mean().round(0).reset_index()
    df_samples_summary.drop(columns=["Run"], inplace=True)
    df_samples_summary.to_csv("Synthetic_Samples_Report.csv", index=False)
    print(f"✅ Sample generation count report saved to: {os.path.abspath('Synthetic_Samples_Report.csv')}")

    print("\n🎉 All experiments successfully concluded! Please check the generated plots and CSV result tables.")


if __name__ == "__main__":
    main(n_runs=5)