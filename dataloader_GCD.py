import numpy as np
import scipy.io
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import torch
from torch.utils.data import Dataset


class Universal_GCD_Dataset(Dataset):
    def __init__(self, X_list, Y, labeled_ratio=0.7, seed=42, use_standard_scaler=False, mode='train', train_ratio=0.8):
        Y = np.array(Y, dtype=np.int32).flatten()
        self.num_view = len(X_list)
        self.class_num = len(np.unique(Y))

        if use_standard_scaler:
            scaler = StandardScaler()
            self.X_list = [scaler.fit_transform(np.array(x, dtype=np.float32)) for x in X_list]
        else:
            self.X_list = [np.array(x, dtype=np.float32) for x in X_list]

        classes = np.unique(Y)
        # 从 0 开始映射标签 (0 到 class_num - 1)，适配 PyTorch 交叉熵
        label_map = {c: i for i, c in enumerate(sorted(classes))}
        Y_adjusted = np.array([label_map[y] for y in Y])

        if self.class_num == 100:
            num_known_classes = 80
        else:
            num_known_classes = int(self.class_num / 2)  # 旧类新类 1:1

        target_classes = np.arange(0, num_known_classes)

        # ------------------- 80/20 训练测试划分逻辑 ------------------------
        np.random.seed(seed)
        train_indices = []
        test_indices = []

        for c in range(self.class_num):
            c_idx = np.where(Y_adjusted == c)[0]
            np.random.shuffle(c_idx)
            split_point = int(len(c_idx) * train_ratio)
            train_indices.extend(c_idx[:split_point])
            test_indices.extend(c_idx[split_point:])

        if mode == 'train':
            self.active_indices = np.array(train_indices)
        elif mode == 'test':
            self.active_indices = np.array(test_indices)
        else:
            raise ValueError("mode must be 'train' or 'test'")

        self.num_sample = len(self.active_indices)
        Y_active = Y_adjusted[self.active_indices]
        Y_new = Y_active.copy()
        self.label_mask = np.zeros(self.num_sample, dtype=bool)

        # ------------------- 伪标签生成逻辑 ------------------------
        if mode == 'train':
            known_mask = np.isin(Y_active, target_classes)
            known_indices = np.where(known_mask)[0]

            num_labeled = int(len(known_indices) * labeled_ratio)
            labeled_indices = np.random.choice(known_indices, size=num_labeled, replace=False)

            self.label_mask[labeled_indices] = True

            unlabeled_indices = np.setdiff1d(np.arange(self.num_sample), labeled_indices)
            Y_new[unlabeled_indices] = -1
            self.num_labeled = len(labeled_indices)
        else:
            Y_new[:] = -1
            self.num_labeled = 0

        self.X = []
        for v in range(self.num_view):
            self.X.append(self.X_list[v][self.active_indices])

        self.Y = Y_new
        self.true_labes = Y_active

    def __len__(self):
        return self.num_sample

    def __getitem__(self, idx):
        views = [torch.from_numpy(self.X[v][idx]) for v in range(self.num_view)]
        return views, self.Y[idx], torch.tensor(idx).long()

    def len_konwnLable_number(self):
        return self.num_labeled


def load_data(dataset, labeled_ratio=0.7, seed=42):
    path = r'E:\mywork\SimpleFramework\data\\'

    use_std_scaler = True if dataset == "Hdigit" else False

    if dataset == "Hdigit":
        data = scipy.io.loadmat(path + 'Hdigit.mat')
        X_list = [data['data'][0][0].astype(np.float32).T, data['data'][0][1].astype(np.float32).T]
        Y = data['truelabel'][0][0]
        dims = [784, 256];
        view = 2;
        class_num = 10

    elif dataset == "BDGP":
        data = scipy.io.loadmat(path + 'BDGP.mat')
        X_list = [data['X1'], data['X2']]
        Y = data['Y'].transpose()
        dims = [1750, 79];
        view = 2;
        class_num = 5

    elif dataset == "MNIST-USPS":
        data = scipy.io.loadmat(path + 'MNIST_USPS.mat')
        X_list = [data['X1'].reshape(-1, 784), data['X2'].reshape(-1, 784)]
        Y = data['Y']
        dims = [784, 784];
        view = 2;
        class_num = 10

    elif dataset == "CCV":
        data1 = MinMaxScaler().fit_transform(np.load(path + 'STIP.npy').astype(np.float32))
        data2 = np.load(path + 'SIFT.npy').astype(np.float32)
        data3 = np.load(path + 'MFCC.npy').astype(np.float32)
        X_list = [data1, data2, data3]
        Y = np.load(path + 'label.npy')
        dims = [5000, 5000, 4000];
        view = 3;
        class_num = 20

    elif dataset == "Fashion":
        data = scipy.io.loadmat(path + 'Fashion.mat')
        X_list = [data['X1'].reshape(-1, 784), data['X2'].reshape(-1, 784), data['X3'].reshape(-1, 784)]
        Y = data['Y']
        dims = [784, 784, 784];
        view = 3;
        class_num = 10

    elif dataset.startswith("Caltech"):
        data = scipy.io.loadmat(path + 'Caltech-5V.mat')
        scaler = MinMaxScaler()
        v1 = scaler.fit_transform(data['X1'].astype(np.float32))
        v2 = scaler.fit_transform(data['X2'].astype(np.float32))
        v3 = scaler.fit_transform(data['X3'].astype(np.float32))
        v4 = scaler.fit_transform(data['X4'].astype(np.float32))
        v5 = scaler.fit_transform(data['X5'].astype(np.float32))
        Y = data['Y'].transpose()
        class_num = 7

        if dataset == "Caltech-2V":
            X_list, dims, view = [v1, v2], [40, 254], 2
        elif dataset == "Caltech-3V":
            X_list, dims, view = [v1, v2, v5], [40, 254, 928], 3
        elif dataset == "Caltech-4V":
            X_list, dims, view = [v1, v2, v5, v4], [40, 254, 928, 512], 4
        elif dataset == "Caltech-5V":
            X_list, dims, view = [v1, v2, v5, v4, v3], [40, 254, 928, 512, 1984], 5

    elif dataset == "Cifar10":
        data = scipy.io.loadmat(path + 'cifar10.mat')
        X_list = [data['data'][0][0].T, data['data'][1][0].T, data['data'][2][0].T]
        Y = data['truelabel'][0][0]
        dims = [512, 2048, 1024];
        view = 3;
        class_num = 10

    elif dataset == "Cifar100":
        data = scipy.io.loadmat(path + 'cifar100.mat')
        X_list = [data['data'][0][0].T, data['data'][1][0].T, data['data'][2][0].T]
        Y = data['truelabel'][0][0]
        dims = [512, 2048, 1024];
        view = 3;
        class_num = 100

    elif dataset == "Synthetic3d":
        data = scipy.io.loadmat(path + 'synthetic3d.mat')
        X_list = [data['X'][0][0], data['X'][1][0], data['X'][2][0]]
        Y = data['Y']
        dims = [3, 3, 3];
        view = 3;
        class_num = 3

    elif dataset == "Prokaryotic":
        data = scipy.io.loadmat(path + 'prokaryotic.mat')
        X_list = [data['X'][0][0], data['X'][1][0], data['X'][2][0]]
        Y = data['Y']
        dims = [438, 3, 393];
        view = 3;
        class_num = 4
    elif dataset == "CUB":
        data = scipy.io.loadmat(path + 'CUB.mat')
        # 常见数据结构 mat['X'][0][0]、[0][1]
        X1 = data['X'][0][0]
        X2 = data['X'][0][1]
        scaler = MinMaxScaler()
        X_list = [scaler.fit_transform(X1.astype(np.float32)),
                  scaler.fit_transform(X2.astype(np.float32))]
        Y = data['gt'].squeeze()
        dims = [X_list[0].shape[1], X_list[1].shape[1]]
        view = 2
        class_num = len(np.unique(Y))

    elif dataset.lower() == "scene15" or dataset == "Scene-15":
        data = scipy.io.loadmat(path + 'Scene-15.mat')
        X1 = data['X'][0][0]
        X2 = data['X'][0][1]
        scaler = MinMaxScaler()
        X_list = [scaler.fit_transform(X1.astype(np.float32)),
                  scaler.fit_transform(X2.astype(np.float32))]
        Y = data['Y'].squeeze()
        dims = [X_list[0].shape[1], X_list[1].shape[1]]
        view = 2
        class_num = len(np.unique(Y))

    elif dataset.upper() == "WIKI":
        data = scipy.io.loadmat(path + 'WIKI.mat')
        X1 = data['Img']
        X2 = data['Txt']
        scaler = MinMaxScaler()
        X_list = [scaler.fit_transform(X1.astype(np.float32)),
                  scaler.fit_transform(X2.astype(np.float32))]
        Y = data['label'].squeeze()
        dims = [X_list[0].shape[1], X_list[1].shape[1]]
        view = 2
        class_num = len(np.unique(Y))

    elif dataset.upper() == "NUS-WIDE":
        data = scipy.io.loadmat(path + 'NUS-WIDE.mat')
        X1 = data['Img']
        X2 = data['Txt']
        scaler = MinMaxScaler()
        X_list = [scaler.fit_transform(X1.astype(np.float32)),
                  scaler.fit_transform(X2.astype(np.float32))]
        Y = data['label'].squeeze()
        dims = [X_list[0].shape[1], X_list[1].shape[1]]
        view = 2
        class_num = len(np.unique(Y))

    elif dataset.lower() == "deep animal" or dataset.lower() == "deep_animal":
        data = scipy.io.loadmat(path + 'Deep_Animal.mat')
        # 视图6和7，一般是 mat['X'][0][5].T  [样本数, 维]
        X1 = data['X'][0][5].T
        X2 = data['X'][0][6].T
        scaler = MinMaxScaler()
        X_list = [scaler.fit_transform(X1.astype(np.float32)),
                  scaler.fit_transform(X2.astype(np.float32))]
        Y = data['gt'].squeeze()
        dims = [X_list[0].shape[1], X_list[1].shape[1]]
        view = 2
        class_num = len(np.unique(Y))
    else:
        raise NotImplementedError

    # 统一生成 train 和 test
    train_dataset = Universal_GCD_Dataset(X_list, Y, labeled_ratio, seed, use_standard_scaler=use_std_scaler,
                                          mode='train')
    test_dataset = Universal_GCD_Dataset(X_list, Y, labeled_ratio, seed, use_standard_scaler=use_std_scaler,
                                         mode='test')

    know_number = train_dataset.len_konwnLable_number()
    return train_dataset, test_dataset, know_number, dims, view, class_num