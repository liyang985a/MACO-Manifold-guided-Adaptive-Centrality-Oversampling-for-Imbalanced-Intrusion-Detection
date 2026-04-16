# ========== Standard Library ==========
import warnings
from collections import Counter, defaultdict
from statistics import mean, stdev

# ========== Third-Party Libraries ==========
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN
from scipy.sparse import SparseEfficiencyWarning
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    cohen_kappa_score, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KDTree, NearestNeighbors
from sklearn.svm import SVC

# ========== PyTorch ==========
import torch.nn as nn
from torch.cuda import is_available

# ========== Custom Modules ==========
from fig import visualize_embedding_comparison, visualize_augmented_tsne
from manifold_mapper5run import neighborhood_Measure_mm
from model2 import measures_of_classify, train_mlp_once

# ========== Warnings ==========
warnings.simplefilter("ignore", SparseEfficiencyWarning)


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
            print(f"🎯 Center sum: {Degree[i, 0]:.4f}, Margin sum: {Degree[i, 1]:.4f}")

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

        # Minority class: retain all samples directly
        if cls_size < min_samples_no_cluster:
            print(f"📌 Class {cls} has few samples ({cls_size}), skipping clustering and filtering, retaining all")
            selected_indices.extend(cls_indices)
            continue

        # Normal class: undergoes filtering and clustering
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
                f"⚠️ Degree is all 0 - mapping method {mapping_type} is invalid for all samples, or neighbor structure is abnormal")

        Degree[:, 0, j] = degree_result[:, 0]
        Degree[:, 1, j] = degree_result[:, 1]

    weighted_cen = np.zeros((r, 1))
    weighted_mar = np.zeros((r, 1))

    # Ensure class order matches the manifold configuration
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
        # Default logic: Use all samples (no filtering)
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
                      min_class_size=5, max_rounds=3, kf=2, iter_num=2):
    if isinstance(data, pd.DataFrame):
        data = data.values

    labels = data[:, -1].astype(int)
    all_precision, all_recall, all_f1, all_acc = [], [], [], []
    final_new_samples = []

    skf = StratifiedKFold(n_splits=kf, shuffle=True, random_state=42)
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
            y_true, y_pred = train_mlp_once_func(X_d, y_d, X_t, y_t, num_classes, device_id=1)

            cm_before = confusion_matrix(y_true, y_pred, labels=range(num_classes))
            r0, p0, f0, _, acc0 = measures_of_classify(cm_before)
            best_f1, best_acc = f0, acc0
            over_data = d.copy()

            cumulative_new_samples = []

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
                        n_generate_per_sample=2,  # Baseline 2
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
                y_true_temp, y_pred_temp = train_mlp_once_func(X_temp, y_temp, X_t, y_t, num_classes, device_id=1)
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

    if final_new_samples:
        final_new_samples_filtered = [s for s in final_new_samples if s.size > 0]
        if final_new_samples_filtered:
            head_df, total_count = save_final_augmented_data(data, final_new_samples_filtered)
            print("✅ Final augmented data saved as final_multiclass_augmentednsl.csv")
            print(f"📦 Total final training samples: {total_count}")
            print(head_df)

    return mean(all_precision), mean(all_recall), mean(all_f1), float('nan'), mean(all_acc)


def save_final_augmented_data(original_data, final_new_samples, filename="final_multiclass_augmentednsl.csv"):
    print("📦 Saving final augmented training set...")
    print(f"   - Original sample count: {original_data.shape[0]}")

    all_new_samples = np.vstack(final_new_samples)
    print(f"   - Augmented sample count (before merge): {sum([s.shape[0] for s in final_new_samples])}")

    # Merge original and augmented data
    final_data = np.vstack((original_data, all_new_samples))
    print(f"   - Total sample count after merge: {final_data.shape[0]}")

    # Use original column names
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


def main(n_runs=5):
    print(f"\n📊 Executing {n_runs} full MLP & M2GDL experiments, calculating average metrics...\n")

    # Baseline Experiment Variables
    base_precision_list, base_recall_list, base_f1_list, base_acc_list, base_mcc_list, base_gmeans_list = [], [], [], [], [], []
    base_class_metrics = defaultdict(lambda: {"precision": [], "recall": [], "f1-score": []})

    # Augmentation Experiment Variables
    precision_list, recall_list, f1_list, acc_list, mcc_list, gmeans_list = [], [], [], [], [], []
    class_metrics = defaultdict(lambda: {"precision": [], "recall": [], "f1-score": []})
    best_result = {"f1": 0}

    for run in range(n_runs):
        print(f"\n================ 🚀 Experiment Run {run + 1}/{n_runs} =================")
        train = pd.read_csv("multi_train_nsl.csv", index_col=0)
        test = pd.read_csv("multi_test_nsl.csv", index_col=0)

        feature_cols = train.columns[:-1]
        label_col = train.columns[-1]

        X_train = train[feature_cols].values
        y_train = train[label_col].values.astype(int)
        X_test = test[feature_cols].values
        y_test = test[label_col].values.astype(int)

        num_classes = len(np.unique(np.concatenate([y_train, y_test])))

        # 1. Baseline Experiment (No Oversampling)
        print("\n🚩 Baseline Experiment (MLP without Oversampling)...")
        y_true_base, y_pred_base = train_mlp_once(X_train, y_train, X_test, y_test, num_classes, device_id=1)

        base_precision = precision_score(y_test, y_pred_base, average='macro', zero_division=0)
        base_recall = recall_score(y_test, y_pred_base, average='macro', zero_division=0)
        base_f1 = f1_score(y_test, y_pred_base, average='macro', zero_division=0)
        base_acc = accuracy_score(y_test, y_pred_base)
        base_mcc = matthews_corrcoef(y_test, y_pred_base)
        base_gmeans = compute_gmeans(y_test, y_pred_base)

        base_precision_list.append(base_precision)
        base_recall_list.append(base_recall)
        base_f1_list.append(base_f1)
        base_acc_list.append(base_acc)
        base_mcc_list.append(base_mcc)
        base_gmeans_list.append(base_gmeans)

        report_base = classification_report(y_test, y_pred_base, output_dict=True, zero_division=0)
        for label in report_base:
            if label.isdigit():
                base_class_metrics[label]["precision"].append(report_base[label]["precision"])
                base_class_metrics[label]["recall"].append(report_base[label]["recall"])
                base_class_metrics[label]["f1-score"].append(report_base[label]["f1-score"])

        # 2. Augmentation Experiment (M2GDL)
        print("\n🚀 Executing M2GDL Augmentation + MLP Experiment...")
        train_data = np.hstack((X_train, y_train.reshape(-1, 1)))
        _ = over_classify_mlp(train_data, train_mlp_once)

        final_augmented_data = pd.read_csv("final_multiclass_augmentednsl.csv").values
        X_aug, y_aug = final_augmented_data[:, :-1], final_augmented_data[:, -1].astype(int)

        y_true_aug, y_pred_aug = train_mlp_once(X_aug, y_aug, X_test, y_test, num_classes, device_id=1)

        macro_precision = precision_score(y_test, y_pred_aug, average='macro', zero_division=0)
        macro_recall = recall_score(y_test, y_pred_aug, average='macro', zero_division=0)
        macro_f1 = f1_score(y_test, y_pred_aug, average='macro', zero_division=0)
        acc = accuracy_score(y_test, y_pred_aug)
        mcc = matthews_corrcoef(y_test, y_pred_aug)
        gmeans = compute_gmeans(y_test, y_pred_aug)

        print(classification_report(y_test, y_pred_aug, zero_division=0))

        precision_list.append(macro_precision)
        recall_list.append(macro_recall)
        f1_list.append(macro_f1)
        acc_list.append(acc)
        mcc_list.append(mcc)
        gmeans_list.append(gmeans)

        report_dict = classification_report(y_test, y_pred_aug, output_dict=True, zero_division=0)
        for label in report_dict:
            if label.isdigit():
                class_metrics[label]["precision"].append(report_dict[label]["precision"])
                class_metrics[label]["recall"].append(report_dict[label]["recall"])
                class_metrics[label]["f1-score"].append(report_dict[label]["f1-score"])

        if macro_f1 > best_result["f1"]:
            best_result.update({
                "run": run + 1,
                "precision": macro_precision,
                "recall": macro_recall,
                "f1": macro_f1,
                "acc": acc,
                "mcc": mcc,
                "gmeans": gmeans
            })

    def stat_str(vals):
        return f"{mean(vals):.4f} ± {stdev(vals):.4f}"

    # Multiple-run Baseline Statistics
    print("\n📉 ✅ Multi-run Baseline Experiment Statistical Results (No Oversampling):")
    print(f"Macro Precision: {stat_str(base_precision_list)}")
    print(f"Macro Recall   : {stat_str(base_recall_list)}")
    print(f"Macro F1       : {stat_str(base_f1_list)}")
    print(f"Accuracy       : {stat_str(base_acc_list)}")
    print(f"MCC            : {stat_str(base_mcc_list)}")
    print(f"G-means        : {stat_str(base_gmeans_list)}")

    # Multiple-run Augmentation Statistics
    print("\n📊 ✅ Multi-run Experiment Statistical Results (M2GDL Augmented):")
    print(f"Macro Precision: {stat_str(precision_list)}")
    print(f"Macro Recall   : {stat_str(recall_list)}")
    print(f"Macro F1       : {stat_str(f1_list)}")
    print(f"Accuracy       : {stat_str(acc_list)}")
    print(f"MCC            : {stat_str(mcc_list)}")
    print(f"G-means        : {stat_str(gmeans_list)}")

    # Per-class Metrics
    print("\n📊 ✅ Average Per-class Metrics (No Oversampling):")
    for label in sorted(base_class_metrics.keys(), key=lambda x: int(x)):
        p = base_class_metrics[label]["precision"]
        r = base_class_metrics[label]["recall"]
        f = base_class_metrics[label]["f1-score"]
        print(f"Class {label}:")
        print(f"  Precision: {mean(p):.4f} ± {stdev(p):.4f}")
        print(f"  Recall   : {mean(r):.4f} ± {stdev(r):.4f}")
        print(f"  F1-Score : {mean(f):.4f} ± {stdev(f):.4f}")

    print("\n📊 ✅ Average Per-class Metrics (M2GDL Augmented):")
    for label in sorted(class_metrics.keys(), key=lambda x: int(x)):
        p = class_metrics[label]["precision"]
        r = class_metrics[label]["recall"]
        f = class_metrics[label]["f1-score"]
        print(f"Class {label}:")
        print(f"  Precision: {mean(p):.4f} ± {stdev(p):.4f}")
        print(f"  Recall   : {mean(r):.4f} ± {stdev(r):.4f}")
        print(f"  F1-Score : {mean(f):.4f} ± {stdev(f):.4f}")

    # Best Single-run Record
    print("\n🏆 Best Single-run Augmented Experiment Results (Sorted by F1):")
    print(f"Run #{best_result['run']}:")
    print(f"Precision: {best_result['precision']:.4f}")
    print(f"Recall   : {best_result['recall']:.4f}")
    print(f"F1 Score : {best_result['f1']:.4f}")
    print(f"Accuracy : {best_result['acc']:.4f}")
    print(f"MCC      : {best_result['mcc']:.4f}")
    print(f"G-means  : {best_result['gmeans']:.4f}")


if __name__ == "__main__":
    main(n_runs=5)