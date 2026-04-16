# model.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np


class IntrusionMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(IntrusionMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.model(x)

#seed=42

def train_mlp_once(X_train, y_train, X_test, y_test, num_classes, epochs=10, batch_size=128, device_id=0):
    import random
    import numpy as np
    import torch

    def set_seed(seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False




    #set_seed(42)




    # ✅ 设置指定 GPU（新增）
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
        device = torch.device(f"cuda:{device_id}")
    else:
        device = torch.device("cpu")



    print(f"\n🟢 使用设备: {device}")
    print(f"📦 训练样本数: {len(X_train)}, 测试样本数: {len(X_test)}, 类别数: {num_classes}")
    print(f"📈 开始训练 MLP，epoch={epochs}, batch_size={batch_size}")

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    #train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True  # 关键修正
    )


    model = IntrusionMLP(input_dim=X_train.shape[1], num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for xb, yb in train_loader:
            preds = model(xb)
            loss = criterion(preds, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"  🔁 Epoch {epoch + 1}/{epochs} | 平均损失: {avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).argmax(dim=1).cpu().numpy()
        y_true = y_test_tensor.cpu().numpy()

    print("✅ 推理完成，开始计算性能指标...\n")
    return y_true, y_pred



def measures_of_classify(xs):
    """
    支持多分类的性能评估函数。
    xs: 混淆矩阵 (n_classes x n_classes)
    返回: macro_recall, macro_precision, macro_F1, NaN(G-means), accuracy
    """
    xs = np.array(xs)
    n_classes = xs.shape[0]

    recall_list = []
    precision_list = []
    f1_list = []

    for i in range(n_classes):
        TP = xs[i, i]
        FN = np.sum(xs[i, :]) - TP
        FP = np.sum(xs[:, i]) - TP

        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) > 0 else 0.0

        recall_list.append(recall)
        precision_list.append(precision)
        f1_list.append(f1)

    macro_recall = np.mean(recall_list)
    macro_precision = np.mean(precision_list)
    macro_f1 = np.mean(f1_list)

    # 多分类 G-means 通常不定义，设为 NaN 占位（后续如需改为 pairwise G 可以添加）
    G_means = np.nan

    accuracy = np.trace(xs) / np.sum(xs)

    return macro_recall, macro_precision, macro_f1, G_means, accuracy
