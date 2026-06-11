import numpy as np
# import scipy.io as sio
import scipy.io
from sklearn.preprocessing import StandardScaler
import os
import torch

#from NCD_python.SimpleFramework.train import target_classes

#NCD [已知类，未知类]
#GCD 已知类中按比例取有标签的数据  已知类无标签+未知类的就是待发现类

class Hdigit():
    def __init__(self, path):
        data = scipy.io.loadmat(path + 'Hdigit.mat')
        self.Y = data['truelabel'][0][0].astype(np.int32).reshape(10000,)
        self.V1 = data['data'][0][0].T.astype(np.float32)  #(feature * samples)
        self.V2 = data['data'][0][1].T.astype(np.float32)

        # (样本×特征)
        self.V1_T = self.V1.T  # (10000, 784)
        self.V2_T = self.V2.T  # (10000, 256)

        self.X = [self.V1_T, self.V2_T]  # 使用转置后的数据做数据类别划分
        # self.X = [self.V1, self.V2]
        self.dims = [self.V1.shape[0], self.V2.shape[0]]

        # 基本统计信息
        self.num_cluster = len(np.unique(self.Y))
        self.num_view = len(self.X)
        self.num_sample = len(self.Y)

        #将每个视图的特征标准化
        scaler = StandardScaler()
        X_normalized = []
        for v in range(self.num_view):
            X_normalized.append(scaler.fit_transform(self.X[v]))

        #调整标签
        Y_adjusted = self.Y.copy()
        if np.any(Y_adjusted == 0):
            Y_adjusted = Y_adjusted +1
            print("Adjusted labels: adder 1 to all labels")

        #定义已知类别 （新数据的前一半）
        target_classes = np.arange(1, int(self.num_cluster / 2) + 1)  #已知类的类别数范围
        target_indices = np.isin(Y_adjusted, target_classes)  #已知类和对应标签
        non_target_indices = ~target_indices #未知类

        #重新组织标签
        Y_new = np.concatenate([Y_adjusted[target_indices], Y_adjusted[non_target_indices]])

        #已知类标签
        Y_new_known = Y_adjusted[target_indices]
        self.known_lable_number = len(Y_new_known)

        #重新组织特征
        X_new = []
        for v in range(len(X_normalized)):
            X_v = X_normalized[v]
            X_new.append(np.hstack([X_v[:, target_indices], X_v[:, non_target_indices]]))

        self.X = [X_new[0].T, X_new[1].T]  #[视图1矩阵（已知类+未知类），视图2矩阵（已知类+未知类）] 矩阵（样本*特征）->(特征*样本）
        self.Y = Y_new

    def __len__(self):
        return 10000
    # def __getitem__(self, idx):
    #     x1 = self.V1[idx]
    #     x2 = self.V2[idx]
    #     return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()
    # #标签 各个视图的特征

    def __getitem__(self, idx):
        x1 = self.X[0][idx]
        x2 = self.X[1][idx]
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

    def len_konwnLable_number(self):
        return self.known_lable_number

def load_data(dataset):
    if dataset == "Hdigit":
        dataset = Hdigit('./data/')
        know_number = dataset.len_konwnLable_number()
        dims = [784, 256]
        view = 2
        data_size = 10000
        class_num = 10
    else:
        raise NotImplementedError
    return dataset, know_number, dims, view, data_size, class_num


if __name__ == "__main__":
    dataset, know_number, dims, view, data_size, class_num = load_data('Hdigit')

    print(f"\n返回结果:")
    print(f"  know_number: {know_number}")
    print(f"  dims: {dims}")
    print(f"  view: {view}")
    print(f"  data_size: {data_size}")
    print(f"  class_num: {class_num}")

    # 测试获取样本
    print(f"\n测试获取样本:")
    for i in range(3):
        views, label, idx = dataset[i]
        print(f"  样本{i}: view1形状={views[0].shape}, view2形状={views[1].shape}, 标签={label}")

    # 验证数据分布
    print(f"\n验证数据分布:")
    print(f"  前{know_number}个样本是已知类别")

    # 检查前几个样本的标签
    known_labels = set(dataset.Y[:know_number])
    novel_labels = set(dataset.Y[know_number:])

    print(f"  已知类别标签: {sorted(known_labels)}")
    print(f"  未知类别标签: {sorted(novel_labels)}")

    # 检查样本0是否属于已知类别
    sample_label = dataset.Y[0]
    is_known = sample_label <= class_num / 2
    print(f"  样本0的标签={sample_label}, 是否已知类别={is_known}")
