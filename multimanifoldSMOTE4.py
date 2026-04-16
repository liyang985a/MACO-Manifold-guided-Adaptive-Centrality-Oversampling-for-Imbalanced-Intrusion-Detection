import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from scipy.stats import entropy
import time
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, KernelPCA
from sklearn.manifold import Isomap
from sklearn.kernel_approximation import Nystroem
####4.2,4.3
'''


'''

class UnifiedManifoldMapper:
    def __init__(
        self,
        mapping_types=None,
        gamma=1.0,
        max_samples_for_exact=3000,
        dim_ratio=0.5,
        max_dim=30,
        use_nystroem=True
    ):
        self.mapping_types = mapping_types or ['PCA', 'KPCA_rbf', 'KPCA_poly']
        self.gamma = gamma
        self.max_samples_for_exact = max_samples_for_exact
        self.dim_ratio = dim_ratio
        self.max_dim = max_dim
        self.use_nystroem = use_nystroem
        self.models = {}
        self.embedding_dims = {}

    def fit(self, X):
        n_samples, n_features = X.shape
        max_pca_dim = min(n_samples, n_features)
        no_dims = min(int(n_features * self.dim_ratio), self.max_dim, max_pca_dim)

        print(f"🔧 降维设置: dim_ratio={self.dim_ratio}, max_dim={self.max_dim} => 使用维度: {no_dims}")

        for type_mapping in self.mapping_types:
            print(f"⏳ 正在处理映射方法: {type_mapping} ...")
            start = time.time()
            model = self._init_model(type_mapping, n_samples, no_dims)
            try:
                X_mapped = model.fit_transform(X)
                end = time.time()
                self.models[type_mapping] = model
                self.embedding_dims[type_mapping] = X_mapped.shape[1]
                print(f"✅ 映射成功: {type_mapping}, 映射后维度: {X_mapped.shape[1]}, 耗时: {end - start:.2f}秒")
            except Exception as e:
                end = time.time()
                print(f"❌ {type_mapping} 映射失败: {e}，耗时: {end - start:.2f}秒")
                self.models[type_mapping] = None
                self.embedding_dims[type_mapping] = X.shape[1]
        return self

    def transform(self, X):
        transformed = {}
        for type_mapping in self.mapping_types:
            model = self.models.get(type_mapping)
            if model is None:
                print(f"⚠️ {type_mapping} 映射器无效，使用原始特征")
                transformed[type_mapping] = X
            else:
                try:
                    transformed[type_mapping] = model.transform(X)
                except Exception as e:
                    print(f"⚠️ {type_mapping} .transform() 失败，使用原始特征: {e}")
                    transformed[type_mapping] = X
        return transformed

    def _init_model(self, type_mapping, n_samples, no_dims):
        no_dims = max(1, min(no_dims, n_samples - 1))  # 强化安全边界
        if type_mapping == 'PCA':
            return PCA(n_components=no_dims)
        elif type_mapping.startswith('KPCA'):
            kernel = 'rbf' if 'rbf' in type_mapping else 'poly'
            if n_samples > self.max_samples_for_exact and self.use_nystroem:
                print(f"⚠️ 样本数 {n_samples} 超过阈值，使用 Nystroem 近似 KPCA（{type_mapping}）")
                return Nystroem(kernel=kernel, n_components=no_dims, gamma=self.gamma)
            elif kernel == 'rbf':
                return KernelPCA(n_components=no_dims, kernel='rbf', gamma=self.gamma)
            else:  # poly
                return KernelPCA(n_components=no_dims, kernel='poly', degree=3, coef0=1)
        elif type_mapping == 'Isomap':
            return Isomap(n_components=no_dims, n_neighbors=10)
        elif type_mapping == 'SVD':
            return TruncatedSVD(n_components=no_dims)
        elif type_mapping == 'ICA':
            return FastICA(n_components=no_dims, random_state=42)
        else:
            raise ValueError(f"❌ 不支持的映射类型: {type_mapping}")


def normalize(arr):
    arr = np.array(arr)
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-6)

def softmax_with_temperature(x, temperature=1.0):
    x = np.array(x)
    x = (x - np.mean(x)) / (np.std(x) + 1e-6)  # z-score normalization
    x = x / temperature
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def fuse_manifold_scores(overlap, temperature=0.6):
    """融合 overlap 与 lcmc 得分（可选加入 entropy）"""
    norm_overlap = normalize(overlap)
    fused_score = norm_overlap

    return softmax_with_temperature(fused_score, temperature)


def compute_rank_overlap(orig_neighbors, embed_neighbors):
    k = orig_neighbors.shape[1]
    total_shift = 0
    for i in range(orig_neighbors.shape[0]):
        ranks = {idx: rank for rank, idx in enumerate(orig_neighbors[i])}
        shift = 0
        for pos, idx in enumerate(embed_neighbors[i]):
            if idx in ranks:
                shift += abs(ranks[idx] - pos)
            else:
                shift += k  # max penalty
        total_shift += shift / k
    return 1.0 - (total_shift / (orig_neighbors.shape[0] * k))


def compute_nmi_lcmc(X, X_embedded, k=10):
    """评估原始与嵌入空间聚类结构一致性"""
    k = min(k, max(2, len(X) // 2))
    try:
        kmeans1 = KMeans(n_clusters=k, random_state=0).fit(X)
        kmeans2 = KMeans(n_clusters=k, random_state=0).fit(X_embedded)
        return (
            normalized_mutual_info_score(kmeans1.labels_, kmeans2.labels_),
            adjusted_rand_score(kmeans1.labels_, kmeans2.labels_)
        )
    except Exception:
        return 0, 0


def compute_entropy_histogram(X_embedded, bins=30):
    """计算每维直方图熵总和"""
    total_entropy = 0
    for i in range(X_embedded.shape[1]):
        hist, _ = np.histogram(X_embedded[:, i], bins=bins, density=True)
        hist += 1e-8
        total_entropy += entropy(hist)
    return total_entropy


def neighborhood_Measure_mm(data, mapper=None, mode='normal',score_mode='normal'):
    """
    多流形结构保持性评估，返回每类样本的 alpha 权重
    返回：
        - manifold: List[Dict]，包含每类样本对应流形的 alpha 权重
        - all_data_map: List[Dict]，降维后的所有数据
        - mapper: 降维器对象
    """
    labels = data[:, -1]
    classes = np.unique(labels)
    all_X = data[:, :-1]


    if mode == 'raw':
        print("🚫 [Ablation-0] 不使用任何流形映射，仅使用原始特征")

        mapper = UnifiedManifoldMapper(mapping_types=['Raw'])
        mapper.models['Raw'] = None
        mapper.embedding_dims['Raw'] = all_X.shape[1]
        all_data_map = [{'all_x': all_X}]

        manifold = []
        for cls in classes:
            alpha = np.array([1.0])  # 只有一个映射（Raw），权重为1
            manifold.append({'alpha': alpha, 'type': ['Raw']})

        return manifold, all_data_map, mapper


    elif mode == 'pca':
        mapper = UnifiedManifoldMapper(mapping_types=['PCA'])
        mapper.fit(all_X)

    elif mode == 'kpca_rbf':
        mapper = UnifiedManifoldMapper(mapping_types=['KPCA_rbf'], gamma=0.5)
        mapper.fit(all_X)

    elif mode == 'kpca_poly':
        mapper = UnifiedManifoldMapper(mapping_types=['KPCA_poly'], gamma=0.5)
        mapper.fit(all_X)

    else:  # 默认多流形
        if mapper is None:
            mapper = UnifiedManifoldMapper(
                mapping_types=['PCA', 'KPCA_rbf', 'KPCA_poly'],
                gamma=0.5,
                dim_ratio=0.2,
                max_dim=60,
                use_nystroem=True
            )
            mapper.fit(all_X)

    all_data_map = []
    for type_mapping in mapper.mapping_types:
        X_mapped = mapper.transform(all_X)[type_mapping]
        X_mapped[np.isnan(X_mapped)] = 0
        all_data_map.append({'all_x': X_mapped})

    manifold = []
    for cls in classes:
        class_data = data[data[:, -1] == cls]
        X_cls = class_data[:, :-1]
        nc = X_cls.shape[0]

        if nc < 6:
            print(f"⚠️ 类别 {cls} 样本数过少（{nc}），跳过分析")
            alpha = np.ones(len(mapper.mapping_types)) / len(mapper.mapping_types)
            manifold.append({'alpha': alpha, 'type': mapper.mapping_types})
            continue

        # 邻居设定
        k = min(max(4, int(np.sqrt(nc)) + 1), nc - 1)
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_cls)
        _, nbrs_before = nbrs.kneighbors(X_cls)
        nbrs_before = nbrs_before[:, 1:]

        overlap_list, lcmc_list  = [], []
        transformed_cls = mapper.transform(X_cls)

        for type_mapping in mapper.mapping_types:
            mappedX = transformed_cls[type_mapping]
            mappedX[np.isnan(mappedX)] = 0
            try:
                nbrs_embed = NearestNeighbors(n_neighbors=k).fit(mappedX).kneighbors(mappedX, return_distance=False)[:, 1:]
                overlap_score = compute_rank_overlap(nbrs_before, nbrs_embed)
                overlap_list.append(overlap_score)
                print(f"类别 {cls} -> {type_mapping} Overlap: {overlap_score:.3f}")
            except Exception as e:
                print(f"❌ 映射失败: {type_mapping}, 类别 {cls} -> {e}")
                overlap_list.append(0)



        if score_mode == 'none':
            alpha = np.ones(len(mapper.mapping_types)) / len(mapper.mapping_types)
            print(f"📊 [Ablation-评分关闭] 类别 {int(cls)} => 使用等权重: {alpha}")
            manifold.append({'alpha': alpha, 'type': mapper.mapping_types})
            continue

        # 融合得分转 softmax 权重
        final_scores = fuse_manifold_scores(overlap_list)
        final_scores += np.random.normal(0, 1e-6, size=final_scores.shape)
        alpha = final_scores
        print(f"📊 类别 {int(cls)} raw_alpha: {np.round(alpha, 3)}")

        manifold.append({'alpha': alpha, 'type': mapper.mapping_types})

    return manifold, all_data_map, mapper


