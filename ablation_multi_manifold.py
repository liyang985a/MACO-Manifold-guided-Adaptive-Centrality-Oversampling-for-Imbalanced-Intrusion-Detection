import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    matthews_corrcoef, precision_score, recall_score,
    f1_score, accuracy_score
)
from MACO_ablation_experiment_multi_manifold import over_classify_mlp, train_mlp_once  # 替换为你的实际模块路径

# ✅ 加载训练和测试数据
train = pd.read_csv("multi_train.csv", index_col=0)
test = pd.read_csv("multi_test.csv", index_col=0)

# ✅ 可选：抽样（调试用，正式实验请设为较大比例）
#train = train.sample(frac=0.01, random_state=42)

# ✅ 特征与标签
feature_cols = train.columns[:-1]
label_col = train.columns[-1]

X_train = train[feature_cols].values
y_train = train[label_col].values.astype(int)
X_test = test[feature_cols].values
y_test = test[label_col].values.astype(int)
train_data = np.hstack((X_train, y_train.reshape(-1, 1)))

num_classes = len(np.unique(np.concatenate((y_train, y_test))))

# ✅ 定义映射策略
strategies = [
    {"name": "Raw", "mapping_mode": "raw"},
    {"name": "PCA", "mapping_mode": "pca"},
    {"name": "KPCA_rbf", "mapping_mode": "kpca_rbf"},
    {"name": "KPCA_poly", "mapping_mode": "kpca_poly"},
    {"name": "Multi-Manifold", "mapping_mode": "normal"},
]

results = []

for strategy in strategies:
    print(f"\n🚀 Running mapping strategy: {strategy['name']}")

    # ✅ 训练阶段：使用增强方法过采样（CV中的表现可以单独记录）
    over_classify_mlp(
        train_data,
        train_mlp_once,
        mapping_mode=strategy["mapping_mode"]
    )

    # ✅ 加载增强后的数据（注意路径要对应）
    final_augmented_data = pd.read_csv("final_multiclass_augmented2.csv").values
    X_aug, y_aug = final_augmented_data[:, :-1], final_augmented_data[:, -1].astype(int)

    # ✅ 使用增强数据重新训练，并在原始测试集上评估
    y_true, y_pred = train_mlp_once(X_aug, y_aug, X_test, y_test, num_classes, device_id=0)

    # ✅ 计算指标（全部基于真实测试集）
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    gmeans = np.sqrt(precision * recall)

    results.append({
        "Mapping": strategy["name"],
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy,
        "MCC": mcc,
        "G-means": gmeans
    })

# ✅ 保存结果为 CSV（用于论文图表生成）
df = pd.DataFrame(results)
df.to_csv("manifold_ablation_results.csv", index=False)
print("\n📄 Saved: manifold_ablation_results.csv")
