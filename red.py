import pandas as pd
import numpy as np

# 定义数据集的文件路径字典
datasets = {
    "UNSW-NB15": {
        "train": "multi_train.csv",
        "test": "multi_test.csv"
    },
    "NSL-KDD": {
        "train": "multi_train_nsl.csv",
        "test": "multi_test_nsl.csv"
    },
    "CICIDS2017": {
        "train": "cicids2017_multi_train.csv",
        "test": "cicids2017_multi_test.csv"
    }
}


def analyze_dataset_imbalance(dataset_name, train_path):
    try:
        # 读取训练集数据
        train = pd.read_csv(train_path, index_col=0)

        # 提取标签列 (最后一列)
        label_col = train.columns[-1]
        y_train = train[label_col].values.astype(int)

        # 统计各个类别的数量
        unique_classes, counts = np.unique(y_train, return_counts=True)
        class_distribution = dict(zip(unique_classes, counts))

        # 计算多数类和少数类
        max_class = max(class_distribution, key=class_distribution.get)
        min_class = min(class_distribution, key=class_distribution.get)
        max_count = class_distribution[max_class]
        min_count = class_distribution[min_class]

        # 计算不平衡比例 (最大类 : 最小类)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

        # 打印统计结果
        print("=" * 40)
        print(f"Dataset: {dataset_name} (Training Set)")
        print("=" * 40)
        print(f"Total Samples: {len(y_train)}")
        print(f"Class Distribution: {class_distribution}")
        print(f"Majority Class: {max_class} (Count: {max_count})")
        print(f"Minority Class: {min_class} (Count: {min_count})")
        print(f"Imbalance Ratio (Majority/Minority): {imbalance_ratio:.2f} : 1")

        # 统计占比小于 1% 的类别数量
        rare_classes = {k: v for k, v in class_distribution.items() if v / len(y_train) < 0.01}
        if rare_classes:
            rare_percentages = {k: f"{(v / len(y_train) * 100):.4f}%" for k, v in rare_classes.items()}
            print(f"Classes making up less than 1%: {rare_percentages}")
        else:
            print("No class makes up less than 1% of the training set.")
        print("\n")

    except FileNotFoundError:
        print(f"Error: Could not find file {train_path} for {dataset_name}\n")


# 循环分析所有数据集
for name, paths in datasets.items():
    analyze_dataset_imbalance(name, paths["train"])