from  torch import nn
from torch.nn.functional import  normalize
import torch

# Encoder && decoder
class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )
    def forward(self, x):
        return self.decoder(x)

class NCDCL(nn.Module):
    def __init__(self, view, input_size, low_feature_dim, high_feature_dim, num_classes, device):
        super(NCDCL, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], low_feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], low_feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        #特定视图
        self.Specific_view = nn.Sequential(
            nn.Linear(low_feature_dim, high_feature_dim),
        )
        #共识
        self.Common_view = nn.Sequential(
            nn.Linear(low_feature_dim * view, high_feature_dim),
        )
        self.view = view

        # ==========================================
        # 新增：ProtoGCD 的原型层 (Prototype Layer)
        # 这是一个无偏置、权重归一化的线性层，权重矩阵表示各类别的原型中心
        # ==========================================
        self.prototype_layer = nn.utils.weight_norm(nn.Linear(high_feature_dim, num_classes, bias=False))
        # 将归一化的缩放因子固定为1，不参与梯度更新
        self.prototype_layer.weight_g.data.fill_(1)
        self.prototype_layer.weight_g.requires_grad = False

    #fusion

    def forward(self, xs):
        xrs = []
        zs = []
        hs = []

        # 1. 提取各个视图的独立特征和重建输出
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = normalize(self.Specific_view(z), dim=1)
            xr = self.decoders[v](z)
            hs.append(h)
            zs.append(z)
            xrs.append(xr)

        # ==========================================
        # 2. 融合多视图特征 (作为原型分类的输入)
        # ==========================================
        commonz_cat = torch.cat(zs, 1)
        # 输出 L2 归一化后的融合特征，这与 ProtoGCD 在超球面上建模的思想一致
        commonz = normalize(self.Common_view(commonz_cat), dim=1)

        # 3. 计算各个样本属于每个类别的 logits (余弦相似度)
        logits = self.prototype_layer(commonz)

        # 4. 获取 L2 归一化后的原型权重 (用于后续计算原型推断/分离损失)
        prototypes = self.prototype_layer.weight_v.clone()
        normed_prototypes = normalize(prototypes, dim=-1, p=2)

        return xrs, zs, hs, commonz, logits, normed_prototypes

    def NCDF(self, xs):
        zs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            zs.append(z)

        commonz = torch.cat(zs, 1)

        commonz = normalize(self.Common_view(commonz), dim=1)
        return commonz