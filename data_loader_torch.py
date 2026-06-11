import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
import os
import torch



class DataLoader:
    def __init__(self, data_path, dataset_name):
        """
        多视图数据加载器

        Args:
            data_path: 数据文件路径
            dataset_name: 数据集名称
        """
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.X = None
        self.Y = None
        self.X_processed = None
        self.Y_processed = None
        self.G_l = None
        self.label_number = None
        self.num_cluster = None
        self.num_view = None
        self.num_sample = None

    def load_data(self, to_tensor=False, device='cpu'):
        """
        加载并预处理数据

        Args:
            to_tensor: 是否转换为PyTorch tensor
            device: 目标设备

        Returns:
            dict: 包含处理后的数据和元信息
        """
        # 构建完整文件路径
        file_path = os.path.join(self.data_path, f"{self.dataset_name}.mat")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        # 加载MAT文件
        data = sio.loadmat(file_path)

        # 提取数据
        self.V1 = data['X1'].T.astype(np.float32)
        self.V2 = data['X2'].T.astype(np.float32)  #(特征*样本）
        self.Y = data['Y'].flatten().astype(np.int32)
        self.X = [self.V1, self.V2]
        self.dims = [self.V1.shape[0], self.V2.shape[0]]

        # 基本统计信息
        self.num_cluster = len(np.unique(self.Y))
        self.num_view = len(self.X)
        self.num_sample = len(self.Y)

        print(f"Dataset: {self.dataset_name}")
        print(f"Samples: {self.num_sample}, Views: {self.num_view}, Clusters: {self.num_cluster}")
        print(f"Dims:{self.dims}")

        return self._preprocess_data(to_tensor, device)

    def _preprocess_data(self, to_tensor=False, device='cpu'):
        """
        数据预处理

        Returns:
            dict: 处理后的数据
        """
        # 标准化特征
        scaler = StandardScaler()
        X_normalized = []
        for v in range(self.num_view):
            X_normalized.append(scaler.fit_transform(self.X[v]))

        # 调整标签（如果需要）
        Y_adjusted = self.Y.copy()
        if np.any(Y_adjusted == 0):
            Y_adjusted = Y_adjusted + 1
            print("Adjusted labels: added 1 to all labels")

        # 按目标类别重新组织数据
        processed_data = self._reorganize_by_classes(X_normalized, Y_adjusted, to_tensor, device)

        return processed_data

    def _reorganize_by_classes(self, X, Y, to_tensor=False, device='cpu'):
        """
        按已知类和新类重新组织数据

        Args:
            X: 标准化后的特征列表
            Y: 调整后的标签
            to_tensor: 是否转换为tensor
            device: 目标设备

        Returns:
            dict: 重新组织后的数据
        """
        # 定义已知类（前一半类别）
        target_classes = np.arange(1, int(self.num_cluster / 2) + 1)
        target_indices = np.isin(Y, target_classes)
        non_target_indices = ~target_indices

        # 重新组织标签
        Y_new = np.concatenate([Y[target_indices], Y[non_target_indices]])

        # 已知类标签
        Gl = Y[target_indices]
        self.label_number = len(Gl)

        # 创建已知类的one-hot编码
        self.G_l = np.zeros((self.label_number, self.num_cluster))
        self.G_l[np.arange(self.label_number), Gl.astype(int) - 1] = 1

        # 重新组织特征
        X_new = []
        for v in range(len(X)):
            X_v = X[v]
            X_new.append(np.hstack([X_v[:, target_indices], X_v[:, non_target_indices]]))

        # 转换为tensor（如果需要）
        if to_tensor:
            X_new = [torch.from_numpy(x).float().to(device) for x in X_new]
            Y_new = torch.from_numpy(Y_new).long().to(device)
            G_l_tensor = torch.from_numpy(self.G_l).float().to(device)
        else:
            G_l_tensor = self.G_l

        # 更新处理后的数据
        self.X_processed = X_new
        self.Y_processed = Y_new

        print(f"Known classes: {target_classes.tolist()}")
        print(f"Known samples: {self.label_number}, Novel samples: {len(Y_new) - self.label_number}")

        return {
            'X': X_new,
            'Y': Y_new,
            'G_l': G_l_tensor,
            'label_number': self.label_number,
            'num_cluster': self.num_cluster,
            'num_view': self.num_view,
            'num_sample': len(Y_new),
            'target_classes': target_classes,
            'dims': self.dims,
            'dataset_name': self.dataset_name
        }

    def get_data_statistics(self):
        """
        获取数据统计信息

        Returns:
            dict: 数据统计信息
        """
        if self.X_processed is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # 检查是否为tensor
        if torch.is_tensor(self.X_processed[0]):
            feature_dims = [X.shape[0] for X in self.X_processed]
        else:
            feature_dims = [X.shape[0] for X in self.X_processed]

        stats = {
            'dataset_name': self.dataset_name,
            'total_samples': self.num_sample,
            'views': self.num_view,
            'clusters': self.num_cluster,
            'known_samples': self.label_number,
            'novel_samples': self.num_sample - self.label_number,
            'feature_dims': feature_dims,
            'known_classes': list(range(1, int(self.num_cluster / 2) + 1)),
            'novel_classes': list(range(int(self.num_cluster / 2) + 1, self.num_cluster + 1))
        }

        return stats

    def get_view_data(self, view_index):
        """
        获取特定视图的数据

        Args:
            view_index: 视图索引

        Returns:
            tuple: (特征矩阵, 特征维度)
        """
        if self.X_processed is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if view_index < 0 or view_index >= self.num_view:
            raise ValueError(f"View index must be between 0 and {self.num_view - 1}")

        return self.X_processed[view_index], self.X_processed[view_index].shape[0]


def load_multiple_datasets(data_path, dataset_names, to_tensor=False, device='cpu'):
    """
    批量加载多个数据集

    Args:
        data_path: 数据路径
        dataset_names: 数据集名称列表
        to_tensor: 是否转换为tensor
        device: 目标设备

    Returns:
        dict: 所有加载的数据集
    """
    datasets = {}

    for name in dataset_names:
        try:
            loader = DataLoader(data_path, name)
            data = loader.load_data(to_tensor=to_tensor, device=device)
            stats = loader.get_data_statistics()
            datasets[name] = {
                'data': data,
                'statistics': stats,
                'loader': loader
            }
            print(f"Successfully loaded: {name}")
        except Exception as e:
            print(f"Failed to load {name}: {str(e)}")

    return datasets

# 使用示例
if __name__ == "__main__":
    # 示例1: 加载单个数据集
    data_path = "../data/"
    dataset_name = "BDGP"

    loader = DataLoader(data_path, dataset_name)
    processed_data = loader.load_data()
    stats = loader.get_data_statistics()

    print("\nData Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")