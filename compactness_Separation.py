import torch
import numpy as np
from network import NCDCL
from dataloader_GCD import load_data
import torch.nn.functional as F


# ==========================================
# 1. 严格按照 ProtoGCD 公式重构的聚类质量评估函数
# ==========================================
def compute_all_metrics(features, labels, class_num, num_known_classes):
    """
    严格按照 ProtoGCD 公式 (26) 和 (27) 计算：
    - Compactness (All, Old, New)
    - Separation (Global)
    """
    features = features.cpu()
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels)
    labels = labels.cpu()

    mu_dict = {}
    compactness_dict = {}

    # --- 步骤 1: 计算所有类的类中心 (mu_k) 和单类紧凑度 ---
    for k in range(class_num):
        mask = (labels == k)
        if mask.sum() == 0:
            continue

        # 提取第 k 类的所有特征
        z_k = features[mask]

        # 计算类均值并进行 L2 归一化 (对应公式中的 \bar{\mu}_k)
        mu_k = z_k.mean(dim=0)
        mu_k = F.normalize(mu_k, p=2, dim=0)
        mu_dict[k] = mu_k

        # 计算该类的内部紧凑度 (公式 26 内层求和)
        sims = torch.matmul(z_k, mu_k)
        compactness_dict[k] = sims.mean().item()

    valid_classes = list(mu_dict.keys())
    if len(valid_classes) == 0:
        return 0.0, 0.0, 0.0, 0.0

    # --- 步骤 2: 计算 Compactness (All, Old, New) (公式 26 外层平均) ---
    old_comps = [compactness_dict[k] for k in valid_classes if k < num_known_classes]
    new_comps = [compactness_dict[k] for k in valid_classes if k >= num_known_classes]
    all_comps = list(compactness_dict.values())

    comp_old = sum(old_comps) / len(old_comps) if len(old_comps) > 0 else 0.0
    comp_new = sum(new_comps) / len(new_comps) if len(new_comps) > 0 else 0.0
    comp_all = sum(all_comps) / len(all_comps) if len(all_comps) > 0 else 0.0

    # --- 步骤 3: 计算全局 Separation (公式 27) ---
    if len(valid_classes) > 1:
        mu_tensor = torch.stack([mu_dict[k] for k in valid_classes])
        # 计算类别中心两两之间的余弦相似度矩阵
        sim_matrix = torch.matmul(mu_tensor, mu_tensor.T)
        # 排除对角线（自己和自己的相似度）
        mask = torch.ones_like(sim_matrix) - torch.eye(len(valid_classes))
        # 求解非对角线元素的平均值
        separation = (sim_matrix * mask).sum().item() / (len(valid_classes) * (len(valid_classes) - 1))
    else:
        separation = 0.0

    return comp_all, comp_old, comp_new, separation


# ==========================================
# 2. 主函数：加载最佳模型并评估
# ==========================================
def main():
    # --- 参数设置 ---
    dataset_name = 'MNIST-USPS'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # --- 1. 加载测试数据 ---
    print(f"正在加载 {dataset_name} 数据集...")

    _, test_dataset, _, dims, view, class_num = load_data(dataset_name)

    if dataset_name == "Cifar100":
        num_known_classes = 80
    else:
        num_known_classes = int(class_num / 2)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    # --- 2. 初始化网络结构 ---
    print("正在初始化 SimpleFramework (NCDCL) 网络...")
    model = NCDCL(view, dims, 512, 128, class_num, device).to(device)

    # --- 3. 加载最优权重 ---
    model_path = f'./models/{dataset_name}_best.pth'
    print(f"正在加载预训练权重: {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    all_features = []

    # --- 4. 提取特征 ---
    print("正在提取测试集融合特征 (commonz)...")
    with torch.no_grad():
        for xs, _, _ in test_loader:
            for v in range(view):
                xs[v] = xs[v].to(device)

            # 正向传播，提取出 commonz (融合层特征)
            *_, commonz, _, _ = model(xs)
            all_features.append(commonz.cpu())

    eval_features = torch.cat(all_features)
    # 从 dataset 底层属性中提取真实的 Ground-Truth 标签
    eval_targets = torch.tensor(test_dataset.true_labes)

    # --- 5. 一键计算所有指标 ---
    comp_all, comp_old, comp_new, sep_global = compute_all_metrics(
        eval_features,
        eval_targets,
        class_num,
        num_known_classes
    )

    # --- 6. 打印最终结果 (对齐论文 Table 13 格式) ---
    print("\n" + "=" * 55)
    print(f"=== SimpleFramework Cluster Metrics ({dataset_name}) ===")
    print("-" * 55)
    print(f"Compactness ↑ (All / Old / New) : {comp_all:.4f} / {comp_old:.4f} / {comp_new:.4f}")
    print(f"Separation  ↓ (Global)     : {sep_global:.4f}")
    print("=" * 55 + "\n")


if __name__ == '__main__':
    main()