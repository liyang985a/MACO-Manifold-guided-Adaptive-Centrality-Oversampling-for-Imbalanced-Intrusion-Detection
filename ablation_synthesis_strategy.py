import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import matthews_corrcoef


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
        f1_score, cohen_kappa_score, matthews_corrcoef, balanced_accuracy_score

# ✅ 引入你的函数（确保 import 正确）
from MACO_ablation_fusion import over_classify_mlp, train_mlp_once  # 替换为你实际的模块名

# ✅ 加载数据
train = pd.read_csv("multi_train.csv", index_col=0)
test = pd.read_csv("multi_test.csv", index_col=0)
#train = train.sample(frac=0.20, random_state=42)

feature_cols = train.columns[:-1]
label_col = train.columns[-1]
X_train = train[feature_cols].values
y_train = train[label_col].values.astype(int)
X_test = test[feature_cols].values
y_test = test[label_col].values.astype(int)

train_data = np.hstack((X_train, y_train.reshape(-1, 1)))

strategies = [
    {"name": "Dict_only", "enable_smote": False, "smote_ratio": 0.8},
    {"name": "SMOTE_only", "enable_smote": True, "smote_ratio": 1.0},
    {"name": "SMOTE[0.8]", "enable_smote": True, "smote_ratio": 0.8},
    {"name": "SMOTE[0.3]", "enable_smote": True, "smote_ratio": 0.3},
    {"name": "SMOTE[0.5]", "enable_smote": True, "smote_ratio": 0.5},

]

results = []
num_classes = len(np.unique(np.concatenate((y_train, y_test))))

for strategy in strategies:
    print(f"\n🚀 当前策略：{strategy['name']}")
    over_classify_mlp(
        train_data,
        train_mlp_once,
        smote_ratio=strategy["smote_ratio"],
        enable_smote=strategy["enable_smote"]
    )

    # ✅ 载入增强后的训练集（注意路径匹配）
    final_augmented_data = pd.read_csv("final_multiclass_augmented.csv").values
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
        "Accuracy": accuracy,  # ✅ 现在是来自测试集的指标
        "MCC": mcc,
        "G-means": gmeans

    })

# ✅ 转为 DataFrame 并保存
df = pd.DataFrame(results)
df.to_csv("synthesis_strategy_ablation.csv", index=False)
print("\n✅ 所有策略运行完毕，结果保存为 synthesis_strategy_ablation.csv")
