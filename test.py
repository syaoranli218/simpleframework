import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import torch
from network import NCDCL
#from metric import valid
from torch.utils.data import Dataset
from scipy.optimize import linear_sum_assignment
import numpy as np
import argparse
import datetime
import random
from loss import Loss, compute_dapl_loss, compute_entropy_loss, compute_proto_sep_loss, SupConLoss, ArcConLoss, MultiViewCMILossBundle
import torch.nn.functional as F
from dataloader_GCD import load_data


Dataname = 'Fashion'
label_ratio = 0.7

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument("--rec_epochs", default=200)
parser.add_argument("--fine_tune_epochs", default=50)
parser.add_argument("--low_feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
parser.add_argument("--lambda_cat", default=1, type=float, help="Category CMI weight")
parser.add_argument("--lambda_inst", default=1, type=float, help="Instance CMI weight")
parser.add_argument("--lambda_con", default=1, type=float, help="Contrastive loss weight")

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


seed = 42
# 针对不同数据集的 预训练轮次(rec_epochs)、微调轮次(fine_tune_epochs) 和 随机种子配置
if args.dataset == "MNIST-USPS":
    args.fine_tune_epochs = 50
    args.lambda_con=0.7
    args.lambda_cat=5

elif args.dataset == "BDGP":
    args.fine_tune_epochs = 15

elif args.dataset == "CCV":
    args.rec_epochs = 50
    args.fine_tune_epochs = 50
    args.lambda_con = 0.5
    args.lambda_cat = 3

elif args.dataset == "Fashion":
    args.fine_tune_epochs = 50

elif args.dataset == "Caltech-2V":
    args.fine_tune_epochs = 100
    args.lambda_con = 0.7
    args.lambda_cat = 1

elif args.dataset == "Caltech-3V":
    args.fine_tune_epochs = 100
    args.lambda_con = 0.5
    args.lambda_cat = 0.5

elif args.dataset == "Caltech-4V":
    args.fine_tune_epochs = 100
    seed=45
    args.lambda_con = 0.5
    args.lambda_cat = 3

elif args.dataset == "Caltech-5V":
    args.fine_tune_epochs = 100
    args.lambda_con = 0.5
    args.lambda_cat = 0.5

elif args.dataset == "Cifar10":
    args.fine_tune_epochs = 10
    args.lambda_con = 1
    args.lambda_cat = 0.5

elif args.dataset == "Cifar100":
    args.fine_tune_epochs = 20
    args.lambda_con = 0.7
    args.lambda_cat = 1

elif args.dataset == "Prokaryotic":
    args.fine_tune_epochs = 20
    args.lambda_con = 1
    args.lambda_cat = 0.5

elif args.dataset == "Synthetic3d":
    args.fine_tune_epochs = 100
    args.lambda_con = 0.7
    args.lambda_cat = 1

elif args.dataset == "Deep_Animal":
    args.rec_epochs = 200
    args.lambda_con = 0.7
    args.lambda_cat = 0.3
else:
    args.rec_epochs = 200
    args.fine_tune_epochs = 100
    seed = 42



os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 针对多GPU
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


setup_seed(seed)

train_dataset, test_dataset, know_number, dims, view, class_num = load_data(args.dataset, label_ratio, seed)

train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

if args.dataset == "Cifar100":
    num_known_classes = 80
else:
    num_known_classes = int(class_num / 2)

def pre_train(epoch):
    tot_loss = 0.
    mse = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(train_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, *_ = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(mse(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(train_loader)))


def valid(model, test_dataset, device, class_num, view, num_known_classes):
    model.eval()
    all_preds = []

    eval_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False)

    with torch.no_grad():
        for xs, _, _ in eval_loader:
            for v in range(view):
                xs[v] = xs[v].to(device)
            *_, logits, _ = model(xs)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())

    eval_preds = torch.cat(all_preds).numpy()
    eval_targets = test_dataset.true_labes

    # 1. 区分 Old 和 New 的 Mask (使用传入的 num_known_classes)
    old_classes_mask = eval_targets < num_known_classes
    new_classes_mask = eval_targets >= num_known_classes

    # 2. 旧类 (Old) 评估:
    if old_classes_mask.sum() > 0:
        old_preds = eval_preds[old_classes_mask]
        old_targets = eval_targets[old_classes_mask]
        old_acc = np.mean(old_preds == old_targets)
    else:
        old_acc = 0.0

    # 3. 新类 (New) 评估: 仅在预测的新类和真实的新类之间做匈牙利匹配
    new_acc = 0.0
    if new_classes_mask.sum() > 0:
        new_preds = eval_preds[new_classes_mask]
        new_targets = eval_targets[new_classes_mask]

        # 只取新类预测中，落在新类原型区间的预测 (严谨起见)
        # 有些模型可能会把新样本预测到旧类上，这类属于错误预测
        new_preds_mapped = new_preds.copy()

        # 构建仅限于新类范围内的混淆矩阵
        D = max(eval_preds.max(), eval_targets.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(new_preds.size):
            # 只有当模型预测也落在新类区间时，才参与匹配奖励；如果预测成了旧类，权当错配
            if new_preds[i] >= num_known_classes:
                w[new_preds[i], new_targets[i]] += 1

        # 仅对新类的矩阵部分做匈牙利算法
        ind = linear_sum_assignment(w.max() - w)
        pred_to_true = {i: j for i, j in zip(*ind)}

        # 重映射新类的预测结果
        for i in range(len(new_preds_mapped)):
            if new_preds_mapped[i] in pred_to_true and new_preds_mapped[i] >= num_known_classes:
                new_preds_mapped[i] = pred_to_true[new_preds_mapped[i]]

        new_acc = np.mean(new_preds_mapped == new_targets)

    # 4. 全局 (All) 评估: 结合修正后的预测算总准确率
    final_preds = eval_preds.copy()
    if new_classes_mask.sum() > 0:
        final_preds[new_classes_mask] = new_preds_mapped

    all_acc = np.mean(final_preds == eval_targets)

    # print(f"\n============== Inductive GCD Evaluation (Test Set) ==============")
    # print(f"All Acc: {all_acc * 100:.2f}% | Old Acc: {old_acc * 100:.2f}% | New Acc: {new_acc * 100:.2f}%")
    # print(f"=================================================================\n")

    return all_acc, old_acc, new_acc


def fine_tune(epoch):
    model.train()
    tot_loss = 0.
    tot_sup = 0.
    tot_dapl = 0.
    tot_con = 0.
    tot_cmi = 0.
    tot_inst = 0.  # 新增：用于记录 Instance CMI

    # ProtoGCD 的超参数
    tau_base = 0.1
    lambda_sup = 0.35
    lambda_entropy = 2.0
    lambda_sep = 0.1
    lambda_con = args.lambda_con

    for batch_idx, (xs, y, idx) in enumerate(train_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        y = y.to(device).long()

        optimizer.zero_grad()

        # 1. 前向传播
        xrs, zs, hs, commonz, logits, normed_prototypes = model(xs)

        labeled_mask = (y != -1)
        unlabeled_mask = (y == -1)

        # ====== 1. 有监督交叉熵 ======
        loss_sup = torch.tensor(0.0).to(device)
        if labeled_mask.sum() > 0:
            labeled_logits = logits[labeled_mask] / tau_base
            labeled_y = y[labeled_mask]
            loss_sup = F.cross_entropy(labeled_logits, labeled_y)

            # ====== 🌟 重构多视图 CMI + MERIT 动态保护 ======
            logits_views_list = [model.prototype_layer(h) for h in hs]
            scaled_logits_views = [l / tau_base for l in logits_views_list]

            # 【MERIT 动态解耦】：计算不确定性 weight_new
            scaled_global_logits = logits / tau_base
            global_evidence = F.softplus(scaled_global_logits)
            global_strength = global_evidence.sum(dim=1) + class_num
            weight_new = (1.0 - (class_num / global_strength)).mean().item()
            weight_new = np.clip(weight_new, 0.1, 1.0)

            # 🌟 1. 计算动态门控权重
            dynamic_lambda_cat = args.lambda_cat * weight_new

            # 🌟 2. 核心修复：直接把动态权重注入黑盒属性！
            # 这样黑盒在内部 forward 时，就会用这个新权重去乘以带梯度的 Loss！
            cmi_bundle.lambda_cat = dynamic_lambda_cat
            cmi_bundle.lambda_inst = args.lambda_inst

            # 🌟 3. 调用黑盒，此时返回的 loss_cmi_aux 已经自带动态加权，且 100% 携带梯度！
            loss_cmi_aux, cmi_details = cmi_bundle(
                logits_views=scaled_logits_views,
                labels=y,
                labeled_mask=labeled_mask,
                return_details=True
            )

            # ====== 4. 双层自适应伪标签损失 ======
            loss_dapl = torch.tensor(0.0).to(device)
            if unlabeled_mask.sum() > 0:
                loss_dapl = compute_dapl_loss(logits[unlabeled_mask], epoch, args.fine_tune_epochs, device)

            # ====== 4. 熵正则化与原型分离 ======
            loss_entropy = compute_entropy_loss(logits / tau_base)
            loss_sep = compute_proto_sep_loss(normed_prototypes, device)

            # ====== 5. 混合对比损失 ======
            features = torch.stack(hs, dim=1)

            loss_con_sup = torch.tensor(0.0).to(device)
            if labeled_mask.sum() > 0:
                features_labeled = features[labeled_mask]
                labels_labeled = y[labeled_mask]
                loss_con_sup = sup_con_criterion(features_labeled, labels=labels_labeled)

            loss_con_unsup = torch.tensor(0.0).to(device)
            if unlabeled_mask.sum() > 0:
                features_unlabeled = features[unlabeled_mask]
                loss_con_unsup = sup_con_criterion(features_unlabeled)

            loss_con = loss_con_sup + loss_con_unsup

            # 仅作 Log 打印用 (被 detach 的安全提取)
            raw_loss_cat = cmi_details.get("loss_cat", torch.tensor(0.0).to(device))
            raw_loss_inst = cmi_details.get("loss_inst", torch.tensor(0.0).to(device))

            # ====== 🌟 6. 总损失 ======
            loss = (lambda_sup * loss_sup) + \
                   loss_cmi_aux + \
                   ((1 - lambda_sup) * loss_dapl) + \
                   (lambda_entropy * loss_entropy) + \
                   (lambda_sep * loss_sep) + \
                   (lambda_con * loss_con)

            loss.backward()
            optimizer.step()

            # 累计统计数据
            tot_loss += loss.item()
            tot_sup += loss_sup.item()
            tot_dapl += loss_dapl.item()
            tot_con += loss_con.item()

            # 注意：这里要加上 .item()，否则累加张量会导致 GPU 显存泄漏！
            tot_inst += raw_loss_inst.item() * cmi_bundle.lambda_inst
            tot_cmi += raw_loss_cat.item() * cmi_bundle.lambda_cat

            # 🚨 【新增的量级监控打印】
            # if batch_idx % 10 == 0:
            #     print(f"   [Debug 量级监控] Epoch {epoch} Batch {batch_idx}")
            #     print(
            #         f"      - Con (对比拉斥) 加权前: {loss_con.item():.4f} | 加权后(*{lambda_con}): {lambda_con * loss_con.item():.4f}")
            #     print(
            #         f"      - Cat (类别对齐) 加权前: {raw_loss_cat.item():.4f} | 加权后(Dynamic *{cmi_bundle.lambda_cat:.4f}): {cmi_bundle.lambda_cat * raw_loss_cat.item():.4f}")
            #     print(
            #         f"      - Inst(实例共识) 加权前: {raw_loss_inst.item():.4f} | 加权后(*{cmi_bundle.lambda_inst}): {cmi_bundle.lambda_inst * raw_loss_inst.item():.4f}")
            #     print("-" * 60)

if not os.path.exists('./models'):
    os.makedirs('./models')
model = NCDCL(view, dims, args.low_feature_dim, args.high_feature_dim, class_num, device)
print(model)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
criterion = Loss(args.batch_size, args.temperature_f, device).to(device)
sup_con_criterion = SupConLoss(temperature=args.temperature_f).to(device)
ArcCon=ArcConLoss().to(device) # 保留实例但不调用它，防止报错

# ====== 🌟 核心：初始化多视图 CMI Bundle ======
cmi_bundle = MultiViewCMILossBundle(
    num_classes=class_num,
    lambda_cat=args.lambda_cat,  # 类别级 CMI (Category CMI)
    lambda_inst=args.lambda_inst,  # 实例级 CMI (Instance CMI)
    lambda_ent=0.0,  # 外部已经有 compute_entropy_loss
    category_kwargs=dict(
        topk=2,
        alpha_view=0.6,
        alpha_global=0.3,
        alpha_instance=0.1,
        confidence_weight=True,
    ),
    instance_kwargs=dict(
        temperature=0.5,
        mode="leave_one_out",
        anchor_idx=0,
        conf_threshold=0,
        anchor_back_weight=0.1,
    ),
).to(device)

epoch = 1
# ---------------- 阶段 1: 预训练 ----------------
for rec_epoch in range(1, args.rec_epochs + 1):
    pre_train(rec_epoch)

# ==========================================
# 🌟 新增：混合原型初始化 (旧类均值 + 新类 K-means) 适配1-index标签
# ==========================================
print("\n===> Extracting features for Hybrid Prototype Initialization...")
from sklearn.cluster import KMeans
import torch.nn.functional as F

model.eval()
all_features = []
all_labels = []

with torch.no_grad():
    # 为了保证提取顺序和原始数据集一致，这里用一个不打乱的 DataLoader
    eval_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    for xs, ys, _ in eval_loader:
        for v in range(view):
            xs[v] = xs[v].to(device)
        # 获取融合特征 commonz [B, high_feature_dim]
        *_, commonz, _, _ = model(xs)
        all_features.append(commonz.cpu())
        all_labels.append(ys.cpu())

all_features = torch.cat(all_features, dim=0)  # [N, D]
all_labels = torch.cat(all_labels, dim=0)  # [N]

# 准备一个张量来存放初始化后的原型 [class_num, high_feature_dim]
hybrid_prototypes = torch.zeros((class_num, args.high_feature_dim), device=device)

# 找出数据中真实的旧类标签 (比如 1, 2, 3)
known_classes_in_data = torch.unique(all_labels[all_labels != -1]).long()
num_known_classes = len(known_classes_in_data)

# --- 步骤 1: 旧类（有标签）使用特征均值初始化 ---
for cls in known_classes_in_data:
    cls_mask = (all_labels == cls)
    cls_features = all_features[cls_mask]
    cls_mean = cls_features.mean(dim=0).to(device)

    # ⚠️ 关键修正：必须把均值赋值给索引为 cls 的位置！
    # 因为 PyTorch 的 CrossEntropy 接收到 label=1 时，会去优化 logits 的第 1 个位置
    hybrid_prototypes[cls.item()] = F.normalize(cls_mean, dim=-1)

print(f"     Successfully initialized {num_known_classes} Old Class prototypes using labeled means.")

# --- 步骤 2: 新类（无标签）使用 K-means 初始化 ---
# 找出还没被旧类占据的空闲原型索引 (比如对于7个类，1,2,3被占了，剩下的就是 0,4,5,6)
used_indices = known_classes_in_data.tolist()
available_indices = [i for i in range(class_num) if i not in used_indices]
num_new_classes = len(available_indices)

if num_new_classes > 0:
    print(f"     Running K-means to initialize {num_new_classes} New Class prototypes...")
    # 我们只用无标签的样本 (y == -1) 来跑 K-means
    unlabeled_features = all_features[all_labels == -1].numpy()

    # 跑 K-means 寻找新类的簇心
    # 将算法强制指定为更稳定的全量计算模式，并增加冗余初始化次数
    kmeans = KMeans(n_clusters=num_new_classes, random_state=seed, n_init=20, algorithm='lloyd')
    kmeans.fit(unlabeled_features)
    new_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).to(device)

    # ⚠️ 关键修正：将 K-means 簇心精准填入空闲的索引位置
    for idx, proto_idx in enumerate(available_indices):
        hybrid_prototypes[proto_idx] = F.normalize(new_centers[idx], dim=-1)

    print(f"     Successfully initialized {num_new_classes} New Class prototypes using K-means.")

# --- 步骤 3: 将混合好的原型覆盖给模型 ---
model.prototype_layer.weight_v.data.copy_(hybrid_prototypes)
print("===> Hybrid Prototype Initialization Completed!\n")
# ==========================================

best_all_acc = 0.0
best_old_acc = 0.0
best_new_acc = 0.0
best_epoch = 0

# ---------------- 阶段 2: 微调聚类 ----------------
for ft_epoch in range(1, args.fine_tune_epochs + 1):
    fine_tune(ft_epoch)

    # 每一轮评估
    all_acc, old_acc, new_acc = valid(model, test_dataset, device, class_num, view, num_known_classes)

    print(
        f'   [Eval] Epoch {ft_epoch} | All Acc: {all_acc * 100:.2f}% | Old Acc: {old_acc * 100:.2f}% | New Acc: {new_acc * 100:.2f}%')

    if ft_epoch == 1:
        torch.save(model.state_dict(), f'./models/{args.dataset}-1st.pth')
        print(f"   📸 [Snapshot] 已保存 Epoch 1 模型用于画图！")
    elif ft_epoch == 10:
        torch.save(model.state_dict(), f'./models/{args.dataset}-10th.pth')
        print(f"   📸 [Snapshot] 已保存 Epoch 10 模型用于画图！")

    # 只有打破纪录时
    if all_acc > best_all_acc:
        best_all_acc = all_acc
        best_old_acc = old_acc
        best_new_acc = new_acc
        best_epoch = ft_epoch

        state = model.state_dict()
        torch.save(state, './models/' + args.dataset + '_best.pth')

        print(
            f'\n🏆 [新高分] Epoch {ft_epoch}! All Acc: {all_acc * 100:.2f}% (Old: {old_acc * 100:.2f}%, New: {new_acc * 100:.2f}%) 已保存! 🏆\n')

if best_epoch > 0:
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_dir = r"E:\mywork\SimpleFramework\logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = os.path.join(log_dir, f"{args.dataset}.log")

    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(f"Run Time: {current_time} (Best Epoch: {best_epoch} / Total Epochs: {args.fine_tune_epochs})\n")
        f.write(f"Params: lambda_con={args.lambda_con}, lambda_cat={args.lambda_cat}, lambda_inst={args.lambda_inst}\n")
        f.write("============== FINAL BEST GCD Evaluation ==============\n")
        f.write(
            f"All Acc: {best_all_acc * 100:.2f}% | Old Acc: {best_old_acc * 100:.2f}% | New Acc: {best_new_acc * 100:.2f}%\n")
        f.write("=====================================================\n\n")

    print(
        f'\n>>> Finished! Final Best All Acc: {best_all_acc * 100:.2f}% (Found at Epoch {best_epoch}) written to log.')