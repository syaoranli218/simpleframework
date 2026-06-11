import torch
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
import argparse

# 仅导入网络骨架和数据，不导入任何带有训练循环的文件！
from network import NCDCL
from dataloader_GCD import load_data


def visualize_tsne(model, dataloader, device, num_known_classes, view, dataset_name,
                   save_path="tsne_visualization.png"):
    """
    万能版 t-SNE 可视化函数 (支持任意数据集、动态视图、智能调色)
    """
    print(f"\n===> Extracting features for {dataset_name} t-SNE Visualization...")
    model.eval()
    all_features = []
    all_labels = []

    true_labels_array = np.array(dataloader.dataset.true_labes)

    with torch.no_grad():
        for xs, ys, idx in dataloader:
            # 动态适应不同数据集的 view 数量
            for v in range(view):
                xs[v] = xs[v].to(device)

            # 提取特征
            *_, commonz, _, normed_prototypes = model(xs)
            all_features.append(commonz.cpu())

            # 获取真实标签
            batch_true_labels = true_labels_array[idx.numpy()]
            all_labels.append(torch.tensor(batch_true_labels))

    all_features = torch.cat(all_features, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    prototypes = normed_prototypes.cpu().numpy()

    combined_features = np.vstack((all_features, prototypes))

    print("     Running t-SNE (this might take a minute)...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    combined_tsne = tsne.fit_transform(combined_features)

    features_2d = combined_tsne[:len(all_features)]
    prototypes_2d = combined_tsne[len(all_features):]

    # 画布设置：适当拉宽，给右侧图例留出充足空间
    plt.figure(figsize=(14, 10))
    plt.gca().axis('off')  # 彻底去除横纵坐标轴
    sns.set_style("white")

    unique_labels = np.unique(all_labels)
    total_classes = len(unique_labels)

    # ==========================================
    # 🌟 1. 智能颜色多样化 (支持超多类别)
    # ==========================================
    # if total_classes <= 10:
    #     # 如果类别少于等于10类，使用你最喜欢的高级定制色
    #     warm_colors = ['#FF4500', '#FFA500', '#FFD700', '#FF69B4', '#F08080']
    #     cool_colors = ['#4169E1', '#00BFFF', '#32CD32', '#9370DB', '#20B2AA']
    # else:
    #     # 如果是像 CCV 这样20类的，自动从大型色带中采样，防止颜色重复
    #     num_new = total_classes - num_known_classes
    #     warm_colors = [cm.Reds(i) for i in np.linspace(0.5, 0.9, num_known_classes)]
    #     cool_colors = [cm.Blues(i) for i in np.linspace(0.4, 0.9, num_new)]
    num_new = total_classes - num_known_classes
    warm_colors = [cm.Reds(i) for i in np.linspace(0.5, 0.9, num_known_classes)]
    cool_colors = [cm.Blues(i) for i in np.linspace(0.4, 0.9, num_new)]

    # ==========================================
    # 🌟 2. 智能类别名称映射
    # ==========================================
    fashion_classes = {
        0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
        5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'
    }

    # 绘制样本点
    for label in unique_labels:
        idx_mask = (all_labels == label)
        is_old_class = label < num_known_classes

        marker = 'o'

        if is_old_class:
            color = warm_colors[int(label) % len(warm_colors)]
            if dataset_name == "Fashion":
                label_name = fashion_classes.get(int(label), f'Old Class {label}')
            else:
                label_name = f'Old Class {label}'
        else:
            new_idx = int(label - num_known_classes)
            color = cool_colors[new_idx % len(cool_colors)]
            if dataset_name == "Fashion":
                base_name = fashion_classes.get(int(label), f'New Class {label}')
            else:
                base_name = f'New Class {label}'
            label_name = f'{base_name} (novel)'

        # 增大散点
        plt.scatter(features_2d[idx_mask, 0], features_2d[idx_mask, 1],
                    color=color, label=label_name,
                    alpha=0.9, s=40, marker=marker, edgecolors='none')

    # 绘制可学习类原型中心点 (纯红色星星)
    if len(prototypes_2d) > 0:
        plt.scatter(prototypes_2d[:, 0], prototypes_2d[:, 1],
                    c='red', marker='*', s=350, edgecolors='none', label='prototypes', zorder=10)

    # 图例排版
    plt.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', markerscale=1.0,
               frameon=False, fontsize=14, labelspacing=0.8, handletextpad=0.2)

    # 动态标题
    # plt.title(f'ProtoGCD (Ours) - {dataset_name}', fontsize=28, fontweight='bold', loc='left', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', transparent=False, facecolor='white')
    plt.close()
    print(f"===> New t-SNE image saved successfully at: {save_path}\n")


if __name__ == "__main__":
    # 增加命令行传参功能，默认是 Fashion
    parser = argparse.ArgumentParser(description='Visualize t-SNE')
    parser.add_argument('--dataset', default="Cifar10", type=str, help="Dataset to visualize")
    args = parser.parse_args()

    dataset_name = args.dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading dataset: {dataset_name}...")
    train_dataset, test_dataset, _, dims, view, class_num = load_data(dataset_name)

    # 🌟 动态计算 num_known_classes (和 test.py 保持完全一致)
    if dataset_name == "Cifar100":
        num_known_classes = 80
    else:
        num_known_classes = int(class_num / 2)

    eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    # 骨架搭建
    model = NCDCL(view=view, input_size=dims, low_feature_dim=512,
                  high_feature_dim=128, num_classes=class_num, device=device).to(device)

    model_path = f'./models/{dataset_name}_best.pth'

    if os.path.exists(model_path):
        print(f"Found saved model weights at {model_path}! Loading...")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        os.makedirs('./pics', exist_ok=True)
        save_path = f'./pics/{dataset_name}_tsne_universal.png'

        # 将 view 和 dataset_name 也传给可视化函数
        visualize_tsne(model, eval_loader, device, num_known_classes, view, dataset_name, save_path=save_path)
    else:
        print(f"❌ Error: 找不到模型文件 {model_path}。请确保你已经训练并保存过该数据集的模型！")