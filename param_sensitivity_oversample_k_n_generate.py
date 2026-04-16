import pandas as pd
import numpy as np
from sklearn.metrics import (
    matthews_corrcoef, precision_score, recall_score,
    f1_score, accuracy_score
)
from MACO_param_sens_Kn_gen import over_classify_mlp, train_mlp_once

# ✅ 数据加载
train = pd.read_csv("multi_train.csv", index_col=0)
test = pd.read_csv("multi_test.csv", index_col=0)
#train = train.sample(frac=0.01, random_state=42)  # 可选：小样本测试
feature_cols = train.columns[:-1]
label_col = train.columns[-1]

X_train = train[feature_cols].values
y_train = train[label_col].values.astype(int)
X_test = test[feature_cols].values
y_test = test[label_col].values.astype(int)

train_data = np.hstack((X_train, y_train.reshape(-1, 1)))
num_classes = len(np.unique(np.concatenate((y_train, y_test))))

results = []

# ✅ Step 1: Baseline
print("📌 Baseline 实验 (oversample_k=5, n_generate_per_sample=2)")
try:
    over_classify_mlp(
        train_data,
        train_mlp_once,
        mapping_mode="normal",
        k_sim=3,
        k_opp=3,
        top_n_opp_classes=3,
        oversample_k=5,
        n_generate_per_sample=2,
        iter_num=2,
        max_rounds=3
    )

    final_augmented_data = pd.read_csv("final_multiclass_augmented4.csv").values
    X_aug, y_aug = final_augmented_data[:, :-1], final_augmented_data[:, -1].astype(int)

    y_true, y_pred = train_mlp_once(X_aug, y_aug, X_test, y_test, num_classes, device_id=0)

    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    gmeans = np.sqrt(precision * recall)

    results.append({
        "Experiment": "Baseline",
        "Param": "oversample_k=5; n_generate=2",
        "F1": f1,
        "Recall": recall,
        "Precision": precision,
        "ACC": accuracy,
        "MCC": mcc,
        "G-means": gmeans
    })
except Exception as e:
    print(f"❌ Baseline 实验失败: {e}")
    results.append({
        "Experiment": "Baseline",
        "Param": "oversample_k=5; n_generate=2",
        "F1": 0, "Recall": 0, "Precision": 0,
        "ACC": 0, "MCC": 0, "G-means": 0
    })

# ✅ Step 2: oversample_k 敏感性分析
for k_val in [1, 3, 7]:
    print(f"\n🔬 实验 oversample_k = {k_val}")
    try:
        over_classify_mlp(
            train_data,
            train_mlp_once,
            mapping_mode="normal",
            k_sim=3, k_opp=3, top_n_opp_classes=3,
            oversample_k=k_val,
            n_generate_per_sample=2,  # 固定
            iter_num=2, max_rounds=3
        )

        final_augmented_data = pd.read_csv("final_multiclass_augmented4.csv").values
        X_aug, y_aug = final_augmented_data[:, :-1], final_augmented_data[:, -1].astype(int)
        y_true, y_pred = train_mlp_once(X_aug, y_aug, X_test, y_test, num_classes, device_id=0)

        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        acc = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        gmeans = np.sqrt(precision * recall)

        results.append({
            "Experiment": "oversample_k",
            "Param": f"oversample_k={k_val}",
            "F1": f1, "Recall": recall, "Precision": precision,
            "ACC": acc, "MCC": mcc, "G-means": gmeans
        })

    except Exception as e:
        print(f"❌ 错误: {e}")
        results.append({
            "Experiment": "oversample_k",
            "Param": f"oversample_k={k_val}",
            "F1": 0, "Recall": 0, "Precision": 0,
            "ACC": 0, "MCC": 0, "G-means": 0
        })

# ✅ Step 3: n_generate_per_sample 敏感性分析
for n_val in [1, 3, 5]:
    print(f"\n🔬 实验 n_generate_per_sample = {n_val}")
    try:
        over_classify_mlp(
            train_data,
            train_mlp_once,
            mapping_mode="normal",
            k_sim=3, k_opp=3, top_n_opp_classes=3,
            oversample_k=5,  # 固定
            n_generate_per_sample=n_val,
            iter_num=2, max_rounds=3
        )

        final_augmented_data = pd.read_csv("final_multiclass_augmented4.csv").values
        X_aug, y_aug = final_augmented_data[:, :-1], final_augmented_data[:, -1].astype(int)
        y_true, y_pred = train_mlp_once(X_aug, y_aug, X_test, y_test, num_classes, device_id=0)

        precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        acc = accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        gmeans = np.sqrt(precision * recall)

        results.append({
            "Experiment": "n_generate_per_sample",
            "Param": f"n_generate={n_val}",
            "F1": f1, "Recall": recall, "Precision": precision,
            "ACC": acc, "MCC": mcc, "G-means": gmeans
        })

    except Exception as e:
        print(f"❌ 错误: {e}")
        results.append({
            "Experiment": "n_generate_per_sample",
            "Param": f"n_generate={n_val}",
            "F1": 0, "Recall": 0, "Precision": 0,
            "ACC": 0, "MCC": 0, "G-means": 0
        })

# ✅ 保存结果
df = pd.DataFrame(results)
df.to_csv("param_sensitivity_k_n_generate.csv", index=False)
print("\n📄 参数敏感性测试完成，结果已保存至 param_sensitivity_k_n_generate.csv")
