import pandas as pd
import numpy as np
from sklearn.metrics import (
    matthews_corrcoef, precision_score, recall_score,
    f1_score, accuracy_score
)
from MACO_parameter_sensitive_ksimkopptopn import over_classify_mlp, train_mlp_once  # 替换为你的模块路径

# ✅ 加载数据
train = pd.read_csv("multi_train.csv", index_col=0)
test = pd.read_csv("multi_test.csv", index_col=0)
#train = train.sample(frac=0.01, random_state=42)  # ✅ 小样本测试时使用

feature_cols = train.columns[:-1]
label_col = train.columns[-1]

X_train = train[feature_cols].values
y_train = train[label_col].values.astype(int)
X_test = test[feature_cols].values
y_test = test[label_col].values.astype(int)
train_data = np.hstack((X_train, y_train.reshape(-1, 1)))
num_classes = len(np.unique(np.concatenate((y_train, y_test))))

# ✅ 基准值
baseline = {
    "k_sim": 3,
    "k_opp": 3,
    "top_n_opp_classes": 3
}

results = []

# ✅ Step 1：添加 Baseline 实验
print("📌 Baseline 实验 (k_sim=3, k_opp=3, top_n_opp_classes=3)")
try:
    over_classify_mlp(
        train_data,
        train_mlp_once,
        mapping_mode="normal",
        k_sim=baseline["k_sim"],
        k_opp=baseline["k_opp"],
        top_n_opp_classes=baseline["top_n_opp_classes"],
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
        "Param": "k_sim=3; k_opp=3; top_n_opp_classes=3",
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
        "Param": "k_sim=3; k_opp=3; top_n_opp_classes=3",
        "F1": 0,
        "Recall": 0,
        "Precision": 0,
        "ACC": 0,
        "MCC": 0,
        "G-means": 0
    })

# ✅ Step 2：A7~A9 单因子实验
experiments = {
    "A7_k_sim": {
        "param_name": "k_sim",
        "values": [1, 5, 7]
    },
    "A8_k_opp": {
        "param_name": "k_opp",
        "values": [1, 5, 7]
    },
    "A9_top_n_opp_classes": {
        "param_name": "top_n_opp_classes",
        "values": [1, 2, 5]
    }
}

for exp_name, config in experiments.items():
    param_name = config["param_name"]
    param_values = config["values"]

    print(f"\n🔬 开始实验 {exp_name} - 单因子敏感性分析")

    for val in param_values:
        print(f"   ▶️ {param_name} = {val}")

        # 设置当前实验参数
        k_sim = val if param_name == "k_sim" else baseline["k_sim"]
        k_opp = val if param_name == "k_opp" else baseline["k_opp"]
        top_n_opp_classes = val if param_name == "top_n_opp_classes" else baseline["top_n_opp_classes"]

        try:
            over_classify_mlp(
                train_data,
                train_mlp_once,
                mapping_mode="normal",
                k_sim=k_sim,
                k_opp=k_opp,
                top_n_opp_classes=top_n_opp_classes,
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
                "Experiment": exp_name,
                "Param": f"{param_name}={val}",
                "F1": f1,
                "Recall": recall,
                "Precision": precision,
                "ACC": accuracy,
                "MCC": mcc,
                "G-means": gmeans
            })

        except Exception as e:
            print(f"   ❌ 错误: {e}")
            results.append({
                "Experiment": exp_name,
                "Param": f"{param_name}={val}",
                "F1": 0,
                "Recall": 0,
                "Precision": 0,
                "ACC": 0,
                "MCC": 0,
                "G-means": 0
            })

# ✅ 保存全部结果
df = pd.DataFrame(results)
df.to_csv("param_sensitivity_A7_A9.csv", index=False)
print("\n📄 参数敏感性实验完成，结果已保存至: param_sensitivity_A7_A9.csv")
