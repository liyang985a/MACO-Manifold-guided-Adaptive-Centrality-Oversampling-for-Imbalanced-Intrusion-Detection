# ========== Standard Library ==========
import warnings
from collections import Counter
from statistics import mean, stdev

# ========== Third-Party Libraries ==========
import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN

from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding, Isomap
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KDTree, NearestNeighbors
from scipy.spatial import KDTree as SciPyKDTree
from scipy.spatial.distance import pdist
from scipy.sparse import SparseEfficiencyWarning
from sklearn.kernel_approximation import Nystroem

# ========== PyTorch ==========
import torch.nn as nn
from sklearn.svm import SVC
from torch.cuda import is_available

# ========== Custom Modules ==========
from fig import visualize_embedding_comparison, visualize_augmented_tsne
from model import train_mlp_once, measures_of_classify
from multimanifoldSMOTE import neighborhood_Measure_mm

# ========== Warnings ==========
warnings.simplefilter("ignore", SparseEfficiencyWarning)

###4.4，4.5

from sklearn.neighbors import KDTree
import numpy as np



def estimate_local_sigma(x, global_features, global_labels, y, k=5):
    indices = np.where(global_labels == y)[0]
    if len(indices) < 2:
        return 1.0
    tree = KDTree(global_features[indices])
    dists, _ = tree.query(x.reshape(1, -1), k=min(k + 1, len(indices)))
    avg_dist = np.mean(dists[0][1:])  # 排除自己
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
    Degree = np.zeros((r, 2))  # 中心性, 边缘性

    for i in range(r):
        x = features[i]
        y = labels[i]

        sigma = estimate_local_sigma(x, global_features, global_labels, y)

        # 同类邻居
        sim_indices_all = np.where(global_labels == y)[0]
        if len(sim_indices_all) >= 1:
            sim_tree = KDTree(global_features[sim_indices_all])
            _, sim_idx_local = sim_tree.query(x.reshape(1, -1), k=min(k_sim, len(sim_indices_all)))
            sim_indices = sim_indices_all[sim_idx_local[0]]
        else:
            sim_indices = []

        # 异类邻居
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
            print(f"🔍 样本 {i} 类别 {y}：同类邻居数={len(sim_indices)}, 异类邻居数={len(opp_indices)}")

        if len(sim_indices) == 0 and len(opp_indices) == 0:
            if verbose:
                print(f"⚠️ 样本 {i} 无有效邻居，Degree 默认设为0")
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
            print(f"🎯 σ={sigma:.4f}, 距离={np.round(all_distances, 2)}")
            print(f"🎯 权重: {np.round(weights, 4)}")
            print(f"🎯 center sum: {Degree[i, 0]:.4f}, margin sum: {Degree[i, 1]:.4f}")

        if verbose and i < 1:
            print(f"🔍 样本 {i} 类别 {y}：")
            print(f"    同类候选样本数: {len(sim_indices_all)}，选中: {len(sim_indices)}")
            print(
                f"    异类候选类别: {top_opp_classes}，样本数: {len(opp_indices_all) if 'opp_indices_all' in locals() else 0}，选中: {len(opp_indices)}")

        if np.all(weights == 0):
            print(f"⚠️ 样本 {i} 所有邻居权重为 0，可能是 σ={sigma:.4f} 太小或邻居太远")


    return Degree





from sklearn.cluster import KMeans


def select_diverse_samples(features, labels, weights, filter_ratio=0.8, n_clusters=3, min_samples_no_cluster=50):
    selected_indices = []
    classes = np.unique(labels)

    for cls in classes:
        cls_mask = (labels == cls)
        cls_features = features[cls_mask]
        cls_weights = weights[cls_mask]
        cls_indices = np.where(cls_mask)[0]
        cls_size = len(cls_features)

        # ✅ 少数类，直接保留全部样本
        if cls_size < min_samples_no_cluster:
            print(f"📌 类别 {cls} 样本较少（{cls_size}），跳过聚类与筛选，保留全部")
            selected_indices.extend(cls_indices)
            continue

        # ✅ 正常类走筛选和聚类流程
        if cls_size < n_clusters:
            top_n = max(1, int(cls_size * filter_ratio))
            top_idx = np.argsort(-cls_weights)[:top_n]
            selected_indices.extend(cls_indices[top_idx])
            continue

        # 聚类+每簇筛选
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        #kmeans = KMeans(n_clusters=n_clusters, n_init=10)

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







#def over_multi_manifold(data, global_data, mapper=None, min_class_size=5,
#                                use_score_filter=True, filter_ratio=0.8):  # ✅ 添加参数

def over_multi_manifold(data, global_data, mapper=None, min_class_size=5,
                            use_score_filter=True, filter_ratio=0.8,
                            mapping_mode='normal'):  # ✅ 添加 mapping_mode

    labels = data[:, -1]
    classes = np.unique(labels)

    class_counts = {cls: np.sum(labels == cls) for cls in classes}
    if any(v < min_class_size for v in class_counts.values()):
        for cls, cnt in class_counts.items():
            if cnt < min_class_size:
                print(f"⚠️ 类别 {cls} 样本数仅 {cnt}，跳过该类的多流形建模。")
        return np.array([]), np.array([]), np.array([])

    #manifold, all_data_map, mapper = neighborhood_Measure_mm(data, mapper=mapper)
    manifold, all_data_map, mapper = neighborhood_Measure_mm(
        data, mapper=mapper, mode=mapping_mode
    )

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
            k_sim=3,  #基准3
            k_opp=3, #基准3
            top_n_opp_classes=3, #基准3
            verbose=False
        )

        if np.all(degree_result[:, 0] == 0) and np.all(degree_result[:, 1] == 0):
            print(f"⚠️ Degree 全为 0 - 映射方法 {mapping_type} 对所有样本无效，或邻居结构异常")

        Degree[:, 0, j] = degree_result[:, 0]
        Degree[:, 1, j] = degree_result[:, 1]

    weighted_cen = np.zeros((r, 1))
    weighted_mar = np.zeros((r, 1))

    # 确保类别与 manifold 顺序一致
    label_to_manifold_index = {cls: idx for idx, cls in enumerate(classes)}
    for i in range(r):
        cls_label = data[i, -1]
        if cls_label not in label_to_manifold_index:
            continue
        cls_index = label_to_manifold_index[cls_label]  # 正确的 manifold 对应下标

        alpha = manifold[cls_index]['alpha']
        if len(alpha) != num_mappings:
            print(f"⚠️ α 长度与 Degree 映射数不符，跳过样本 {i}")
            continue

        weighted_cen[i, 0] = np.dot(alpha, Degree[i, 0, :])
        weighted_mar[i, 0] = np.dot(alpha, Degree[i, 1, :])

    #weight = weighted_mar - weighted_cen
    weight = Degree[:, 1] / (Degree[:, 0] + 1e-6)  # W = mar / (cen + ε)


    if use_score_filter:
        # mask_marginal = weight[:, 0] > 0
        # weight_marginal = weight[mask_marginal]
        # data_marginal = data[mask_marginal]
        # label_marginal = data_marginal[:, -1]

        # 使用所有数据而不是仅weight>0的子集
        weight_marginal = weight
        data_marginal = data
        label_marginal = data[:, -1]



        if len(weight_marginal) == 0:
            print("⚠️ 无边缘样本可供筛选，返回空")
            return np.array([]), np.array([]), np.array([])

        selected_indices = select_diverse_samples(
            data_marginal[:, :-1],
            label_marginal,
            weight_marginal[:, 0],
            filter_ratio=filter_ratio,
            n_clusters=3,
            min_samples_no_cluster=100  # ✅ 加上这个参数
        )

        sorted_data = data_marginal[selected_indices]
        sorted_weight = weight_marginal[selected_indices]
        sort_label = label_marginal[selected_indices]
        #sorted_indices = np.argsort(-sorted_weight[:, 0])
        print(f"🎯 使用 score filter: weight>0 且 top {filter_ratio * 100:.0f}% ，保留 {len(sorted_weight)} 个样本")


    else:
        # ✅ 默认逻辑：使用所有样本（不筛选）
        sorted_indices = np.argsort(-weight[:, 0])
        sorted_weight = weight[sorted_indices]
        sorted_data = data[sorted_indices, :]
        sort_label = data[sorted_indices, -1]


    #print(f"✅ Degree 合成后（mar - cen）值范围: {weight.min():.4f} ~ {weight.max():.4f}")
    #print(f"📊 Top 10 样本 raw weight: {sorted_weight[:10].flatten()}")
    #print(f"📌 Top 10 样本标签: {sort_label[:10]}")
    #print(f"🎯 Top 10 样本中心性: {weighted_cen[sorted_indices[:10]].flatten()}")
    #print(f"🔍 Top 10 样本边缘性: {weighted_mar[sorted_indices[:10]].flatten()}")

    return sorted_weight, sorted_data, sort_label









###4.6

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist


def average_distance(minority_data):
    if minority_data.shape[0] < 2:
        return 0.0
    distances = pdist(minority_data, metric='euclidean')
    return np.mean(distances)

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

def gradual_overSampling_func_multi_plus(
        sorted_weight,
        sorted_data,
        sort_label,
        target_class,
        alpha=1.0,
        beta=0.1,
        k=5,
        max_iter=20,
        eps=0.005,
        n_generate_per_sample=1,
        smote_ratio=0.5,
        enable_smote=False
):
    r, c = sorted_data.shape
    class_data = sorted_data[sorted_data[:, c - 1] == target_class, :]
    if class_data.shape[0] == 0:
        print(f"⚠️ 类别 {target_class} 无边缘样本，跳过增强。")
        return np.array([]), 1.0  # fallback_ratio=1 表示失败

    print(f"\n✅ [类 {target_class}] 用于过采样的边缘样本数: {class_data.shape[0]}")

    features = np.array(class_data[:, :-1], dtype=np.float64)
    if features.ndim != 2 or features.shape[1] == 0:
        print(f"❌ 特征提取失败: 非法维度 {features.shape}")
        return np.array([]), 1.0

    nc, feature_dim = features.shape
    if nc <= 1:
        print(f"⚠️ 类别 {target_class} 样本过少，无法建立邻居结构。")
        return np.array([]), 1.0

    k_eff = min(k, nc - 1)
    if k_eff < 1:
        print(f"⚠️ 类别 {target_class} 可用样本不足，无法建立邻居结构。")
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
        print(f"⚠️ 类别 {target_class} 无增强样本生成。")
        return np.array([]), 1.0

    Over = np.array(generated_samples)
    label_newdata = np.full((Over.shape[0], 1), target_class, dtype=sorted_data.dtype)
    Over_Sampling_Data = np.hstack((Over, label_newdata))
    Over_Sampling_Data = np.unique(Over_Sampling_Data, axis=0)

    if converge_iters:
        converge_iters = np.array(converge_iters)
        print(f"📊 插值总数: {len(converge_iters)}")
        print(f"📊 平均插值迭代轮数: {np.mean(converge_iters):.2f}，最大迭代数: {np.max(converge_iters)}")
        print(f"📌 Fallback（未收敛）插值样本数: {fallback_count}")
        print(f"📊 插值方式统计：")
        print(f"   ├─ SMOTE 插值数        : {smote_count}")
        print(f"   ├─ 字典插值成功数      : {dict_success_count}")
        print(f"   └─ 字典插值 Fallback数 : {dict_fallback_count}")

    return Over_Sampling_Data





##4.7
#def over_classify_mlp(data, train_mlp_once_func, min_class_size=5, max_rounds=3, kf=2, iter_num=2):
def over_classify_mlp(data, train_mlp_once_func,
                          min_class_size=5, max_rounds=3, kf=2, iter_num=2,
                          mapping_mode='normal'):  # ✅ 添加

    if isinstance(data, pd.DataFrame):
        data = data.values

    labels = data[:, -1].astype(int)
    all_precision, all_recall, all_f1, all_acc = [], [], [], []
    final_new_samples = []

    skf = StratifiedKFold(n_splits=kf, shuffle=True, random_state=42)
    #skf = StratifiedKFold(n_splits=kf, shuffle=True)
    split_indices = list(skf.split(data[:, :-1], labels))

    for j in range(iter_num):
        print(f"\n🔁 第 {j + 1}/{iter_num} 次迭代中...")
        for i, (train_idx, test_idx) in enumerate(split_indices):
            print(f"   ▶️ 第 {i + 1}/{kf} 折交叉验证中...")
            d, t = data[train_idx], data[test_idx]

            label_counts = Counter(d[:, -1])
            counts = np.array(list(label_counts.values()))

            # （基于平均数的60%）
            mean_count = np.mean(counts)
            threshold = 0.6 * mean_count  # 可调参数，控制哪些类算“少数”

            valid_classes = [cls for cls, count in label_counts.items() if count < threshold]
            print(f"      - 当前类别样本数: {dict(label_counts)}")
            print(f"      - 平均样本数: {mean_count:.0f}，少数类阈值: {threshold:.0f}")
            print(f"      - 动态选出的可增强类别: {valid_classes}")

            if not valid_classes:
                print("⚠️ 当前无有效少数类，跳过增强")
                continue

            X_d, y_d = d[:, :-1], d[:, -1].astype(int)
            X_t, y_t = t[:, :-1], t[:, -1].astype(int)
            num_classes = len(np.unique(np.concatenate((y_d, y_t))))
            y_true, y_pred = train_mlp_once_func(X_d, y_d, X_t, y_t, num_classes,device_id=1)

            cm_before = confusion_matrix(y_true, y_pred, labels=range(num_classes))
            r0, p0, f0, _, acc0 = measures_of_classify(cm_before)
            best_f1, best_acc = f0, acc0
            over_data = d.copy()
            best_new_samples = None
            print(f"      📊 原始 F1: {f0:.4f} | 准确率: {acc0:.4f}")

            for round_num in range(1, max_rounds + 1):
                print(f"      🔁 第 {round_num} 轮增强中...")
                new_data_all_classes = []

                for cls in valid_classes:
                    cls_data = over_data[over_data[:, -1] == cls]
                    if cls_data.shape[0] < min_class_size:
                        print(f"        ⚠️ 类别 {cls} 样本数太少，跳过增强")
                        continue

                    #sorted_weight, sorted_data, sort_label = over_multi_manifold(cls_data, d)
                    sorted_weight, sorted_data, sort_label = over_multi_manifold(
                        cls_data, d,
                        mapping_mode=mapping_mode
                    )

                    if sorted_data.ndim != 2 or sorted_data.shape[0] == 0:
                        print(f"        ⚠️ 类别 {cls} 无法执行 over_multi_manifold，跳过增强")
                        continue


                    new_cls_data = gradual_overSampling_func_multi_plus(
                        sorted_weight, sorted_data, sort_label,
                        target_class=cls,
                        n_generate_per_sample=2  # 基准2
                        , enable_smote=True, smote_ratio=0.8
                    )


                    if new_cls_data.size > 0:

                        target_count = int(np.mean(list(label_counts.values())))
                        current_count = label_counts[cls]
                        needed = target_count - current_count
                        if needed <= 0:
                            continue
                        # 控制最大增强量，避免爆炸
                        max_add = min(needed, new_cls_data.shape[0])
                        new_cls_data = new_cls_data[:max_add]  # 或加 random.choice 增加多样性

                        print(f"        ✅ 类别 {cls} 增强样本: {new_cls_data.shape[0]}（原始: {label_counts[cls]}）")
                        new_data_all_classes.append(new_cls_data)
                    else:
                        print(f"        ⚠️ 类别 {cls} 无增强样本生成")

                if not new_data_all_classes:
                    print("        ❌ 本轮无增强样本生成，提前终止")
                    break

                new_samples = np.vstack(new_data_all_classes)
                temp_over_data = np.vstack((over_data, new_samples))
                temp_over_data = np.unique(temp_over_data, axis=0)

                X_temp, y_temp = temp_over_data[:, :-1], temp_over_data[:, -1].astype(int)
                y_true_temp, y_pred_temp = train_mlp_once_func(X_temp, y_temp, X_t, y_t, num_classes,device_id=1)
                cm_after = confusion_matrix(y_true_temp, y_pred_temp, labels=range(num_classes))
                r1, p1, f1, _, acc1 = measures_of_classify(cm_after)

                print(f"        🧪 增强后 F1: {f1:.4f} | 准确率: {acc1:.4f}")

                if f1 >= best_f1:
                    best_f1, best_acc = f1, acc1
                    over_data = temp_over_data
                    best_new_samples = new_samples
                else:
                    print("        ⚠️ 增强无显著提升，停止迭代")
                    break

            if best_new_samples is not None:
                final_new_samples.append(best_new_samples)

            all_precision.append(p1)
            all_recall.append(r1)
            all_f1.append(best_f1)
            all_acc.append(best_acc)
            print(f"   ✅ 折内最佳 F1: {best_f1:.4f} | 准确率: {best_acc:.4f}")

    print("\n✅ 全部交叉验证完成")
    print(f"📌 平均 F1: {mean(all_f1):.4f}, 平均准确率: {mean(all_acc):.4f}")

    # 保存最终增强数据 = 原始样本 + 所有增强样本
    # 在 over_classify_mlp 函数末尾替换保存逻辑为：
    if final_new_samples:
        final_new_samples_filtered = [s for s in final_new_samples if s.size > 0]
        if final_new_samples_filtered:
            head_df, total_count = save_final_augmented_data(data, final_new_samples_filtered)

            print("✅ 最终增强数据已保存为 final_multiclass_augmented2.csv")
            print(f"📦 最终训练样本总数: {total_count}")
            print(head_df)

    return mean(all_precision), mean(all_recall), mean(all_f1), float('nan'), mean(all_acc)


def save_final_augmented_data(original_data, final_new_samples, filename="final_multiclass_augmented2.csv"):
    print("📦 正在保存最终增强训练集...")
    print(f"   - 原始样本数: {original_data.shape[0]}")

    all_new_samples = np.vstack(final_new_samples)
    print(f"   - 增强样本数（合并前）: {sum([s.shape[0] for s in final_new_samples])}")

    # 合并原始数据和增强数据
    final_data = np.vstack((original_data, all_new_samples))
    print(f"   - 合并后总样本数: {final_data.shape[0]}")

    # ✅ 使用原始列名（假设原始是 DataFrame 格式）
    if isinstance(original_data, pd.DataFrame):
        col_names = list(original_data.columns)
    else:
        # 如果传入的是 np.array，则 fallback 为 feature_x
        num_features = final_data.shape[1] - 1
        col_names = [f"feature_{i + 1}" for i in range(num_features)] + ["label"]

    df_save = pd.DataFrame(final_data, columns=col_names)
    df_save.to_csv(filename, index=False)
    print(f"✅ 增强训练集已保存为: {filename}\n")

    return df_save.head(), final_data.shape[0]



from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
        f1_score, cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score




def main():
    # ✅ 1. 加载训练集和测试集（直接跳过首列索引）
    train = pd.read_csv("multi_train.csv", index_col=0)
    test = pd.read_csv("multi_test.csv", index_col=0)
    #train = train.sample(frac=0.2, random_state=42)



    # ✅ 2. 特征与标签分离（推荐列名操作，避免依赖列位置）
    feature_cols = train.columns[:-1]  # 假设最后一列是标签列
    label_col = train.columns[-1]

    X_train = train[feature_cols].values
    y_train = train[label_col].values.astype(int)
    X_test = test[feature_cols].values
    y_test = test[label_col].values.astype(int)

    # ✅ 3. 对照实验：未过采样 MLP
    print("\n🚩 对照实验（未过采样 MLP）...")

    print(f"GPU 可用: {is_available()}")
    num_classes = len(np.unique(np.concatenate((y_train, y_test))))
    y_true, y_pred = train_mlp_once(X_train, y_train, X_test, y_test, num_classes,device_id=0)
    print(classification_report(y_test, y_pred, zero_division=0))

    print(f"MLP（未过采样）精确率 (Macro Precision): {precision_score(y_true, y_pred, average='macro'):.4f}")
    print(f"MLP（未过采样）召回率 (Macro Recall): {recall_score(y_true, y_pred, average='macro'):.4f}")
    print(f"MLP（未过采样）F-measure (Macro F1): {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"MLP（未过采样）准确率 (Accuracy): {accuracy_score(y_true, y_pred):.4f}")

    # ✅ 4. 构建用于 M2GDL 的训练数据（加上标签）
    train_data = np.hstack((X_train, y_train.reshape(-1, 1)))

    # ✅ 5. 执行多类过采样增强 MLP（M2GDL）
    print("\n🚀 执行多类过采样增强后的 MLP 实验（M2GDL）...")
    results = over_classify_mlp(train_data, train_mlp_once)

    (after_Precision, after_Recall, after_F1,
     _, after_accuracy) = results

    print(f"\n✅ 交叉验证：多类过采样 MLP 平均指标（M2GDL）：")
    print(f"增强后精确率 (Macro Precision): {after_Precision:.4f}")
    print(f"增强后召回率 (Macro Recall): {after_Recall:.4f}")
    print(f"增强后F-measure (Macro F1): {after_F1:.4f}")
    print(f"增强后准确率 (Accuracy): {after_accuracy:.4f}")

    # ✅ 6. 使用增强后的数据，在真实测试集上评估 M2GDL

    print("\n🚀 使用 M2GDL 增强数据重新训练，并在真实测试集上评估...")
    final_augmented_data = pd.read_csv("final_multiclass_augmented2.csv").values
    X_aug, y_aug = final_augmented_data[:, :-1], final_augmented_data[:, -1].astype(int)
    y_true, y_pred = train_mlp_once(X_aug, y_aug, X_test, y_test, num_classes, device_id=0)

    # ✅ 样本级评估
    macro_precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)
    g_means = np.sqrt(macro_precision * macro_recall)

    # ✅ 报告
    print("\n🧪 最终评估结果（M2GDL 增强 + MLP）：")
    print(f"  📏 Accuracy                : {accuracy:.4f}")
    print(f"  📏 Balanced Accuracy       : {balanced_acc:.4f}")
    print(f"  📏 Macro Precision         : {macro_precision:.4f}")
    print(f"  📏 Macro Recall            : {macro_recall:.4f}")
    print(f"  📏 Macro F1-score          : {macro_f1:.4f}")
    print(f"  📏 Weighted F1-score       : {weighted_f1:.4f}")
    print(f"  📏 MCC (Matthews CorrCoef) : {mcc:.4f}")
    print(f"  📏 Cohen's Kappa           : {kappa:.4f}")
    print(f"  📏 G-means                 : {g_means:.4f}")

    print("\n📑 分类报告 (Per-class 指标):")
    print(classification_report(y_test, y_pred, zero_division=0))



if __name__ == "__main__":
    main()


