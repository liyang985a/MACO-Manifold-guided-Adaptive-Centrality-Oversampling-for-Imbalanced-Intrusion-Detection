import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef
from MACO_ablation_experiment_score_filtering import over_classify_mlp, train_mlp_once  # 替换为你模块实际名

# ✅ 加载数据
train = pd.read_csv("multi_train.csv", index_col=0)
test = pd.read_csv("multi_test.csv", index_col=0)
#train = train.sample(frac=0.01, random_state=42)  # 测试时可调

feature_cols = train.columns[:-1]
label_col = train.columns[-1]
X_train = train[feature_cols].values
y_train = train[label_col].values.astype(int)
X_test = test[feature_cols].values
y_test = test[label_col].values.astype(int)

train_data = np.hstack((X_train, y_train.reshape(-1, 1)))
num_classes = len(np.unique(np.concatenate((y_train, y_test))))

# ✅ 实验策略（主要对 use_score_filter 做消融）
strategies = [
    {"name": "NoFilter", "use_score_filter": False, "filter_ratio": 1.0},
    {"name": "ScoreFilter[0.8]", "use_score_filter": True, "filter_ratio": 0.8},
    {"name": "ScoreFilter[0.5]", "use_score_filter": True, "filter_ratio": 0.5},
    {"name": "ScoreFilter[0.3]", "use_score_filter": True, "filter_ratio": 0.3},
]


results = []

for strategy in strategies:
    print(f"\n🚀 当前策略: {strategy['name']}")

    # ✅ 运行多流形增强（内部会写出增强数据）
    over_classify_mlp(
        train_data,
        train_mlp_once,
        use_score_filter=strategy["use_score_filter"],
        filter_ratio=strategy["filter_ratio"]
    )

    # ✅ 载入增强后的训练集（注意路径匹配）
    final_augmented_data = pd.read_csv("final_multiclass_augmented3.csv").values
    X_aug, y_aug = final_augmented_data[:, :-1], final_augmented_data[:, -1].astype(int)

    # ✅ 用增强数据重新训练并评估
    y_true, y_pred = train_mlp_once(X_aug, y_aug, X_test, y_test, num_classes, device_id=0)

    # ✅ 计算指标
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    gmeans = np.sqrt(precision * recall)

    results.append({
        "Strategy": strategy["name"],
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": accuracy,
        "MCC": mcc,
        "G-means": gmeans
    })

# ✅ 保存为 CSV
df = pd.DataFrame(results)
df.to_csv("score_filter_ablation_results.csv", index=False)
print("\n📄 已保存：score_filter_ablation_results.csv")
