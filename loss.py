import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcConLoss(nn.Module):
    def __init__(self, temperature=0.5, margin=0.2, eps=1e-7):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.eps = eps

    def _pair_loss(self, z_a, z_b):
        """
        z_a: [B, D]
        z_b: [B, D]
        """
        z_a = F.normalize(z_a, dim=-1)
        z_b = F.normalize(z_b, dim=-1)

        # [B, B]
        similarity_matrix = torch.matmul(z_a, z_b.T)

        # 对角线：正样本
        diagonal_elements = torch.diag(similarity_matrix)

        # 非对角线：负样本
        mask = ~torch.eye(similarity_matrix.size(0), dtype=torch.bool, device=similarity_matrix.device)
        other_elements = similarity_matrix[mask]

        # 防止 acos 数值越界
        diagonal_elements = torch.clamp(diagonal_elements, -1.0 + self.eps, 1.0 - self.eps)
        other_elements = torch.clamp(other_elements, -1.0 + self.eps, 1.0 - self.eps)

        theta_pos = torch.acos(diagonal_elements)
        theta_neg = torch.acos(other_elements)

        numerator = torch.sum(
            torch.exp(torch.cos(theta_pos + self.margin) / self.temperature)
        )

        denominator = numerator + torch.sum(
            torch.exp(torch.cos(theta_neg) / self.temperature)
        )

        loss = torch.log(denominator) - torch.log(numerator)
        return loss

    def forward(self, embs):
        """
        embs: [B, V, D]
        比如 torch.stack([emb1, emb2, emb3], dim=1)
        """
        if embs.dim() != 3:
            raise ValueError(f"embs should be [B, V, D], but got shape {embs.shape}")

        B, V, D = embs.shape

        if V < 2:
            raise ValueError("Need at least 2 views.")

        losses = []
        for a in range(V):
            for b in range(a + 1, V):
                z_a = embs[:, a, :]   # [B, D]
                z_b = embs[:, b, :]   # [B, D]
                losses.append(self._pair_loss(z_a, z_b))

        return torch.stack(losses).mean()
class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    From: https://github.com/HobbitLong/SupContrast"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
class Loss(nn.Module):
    def __init__(self, batch_size, temperature_f, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.temperature_f = temperature_f
        self.device = device
        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask

    def Structure_guided_Contrastive_Loss(self, h_i, h_j, S):
        S_1 = S.repeat(2, 2)
        all_one = torch.ones(self.batch_size*2, self.batch_size*2).to('cuda')
        S_2 = all_one - S_1
        N = 2 * self.batch_size
        h = torch.cat((h_i, h_j), dim=0)
        sim = torch.matmul(h, h.T) / self.temperature_f
        sim1 = torch.multiply(sim, S_2)
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim1[mask].reshape(N, -1)
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss



def compute_dapl_loss(logits, epoch, total_epochs, device):
        """
        双层自适应伪标签损失 (Dual-level Adaptive Pseudo-Labeling, DAPL)
        """
        # ProtoGCD 论文中的超参数
        tau_base = 0.1
        tau_sharp = 0.05

        batch_size = logits.size(0)
        if batch_size == 0:
            return torch.tensor(0.0).to(device)

        with torch.no_grad():
            # 1. 计算样本置信度 (基于 Top-1 和 Top-2 的余弦相似度差距)
            topk_logits, _ = torch.topk(logits, k=2, dim=1)
            top1 = topk_logits[:, 0]
            top2 = topk_logits[:, 1]
            # 公式 (8): exp(top1/tau) / exp(top2/tau) <=> exp((top1 - top2) / tau)
            confidences = torch.exp((top1 - top2) / tau_base)

            # 2. 计算当前 Epoch 的硬标签比例 (Level-2 Adaptivity: 随着训练推进线性增加)
            ratio = min(1.0, epoch / total_epochs)
            num_hard = int(batch_size * ratio)

            # 3. 按置信度对当前批次的样本进行降序排序 (Level-1 Adaptivity)
            _, sorted_idx = torch.sort(confidences, descending=True)
            hard_idx = sorted_idx[:num_hard]
            soft_idx = sorted_idx[num_hard:]

            # 4. 生成自适应伪标签 targets
            targets = torch.zeros_like(logits).to(device)

            # 对高置信度样本：分配 One-hot 硬标签
            if len(hard_idx) > 0:
                hard_preds = torch.argmax(logits[hard_idx], dim=1)
                targets[hard_idx] = F.one_hot(hard_preds, num_classes=logits.size(1)).float()

            # 对低置信度样本：分配 Sharpen 后的软标签
            if len(soft_idx) > 0:
                targets[soft_idx] = F.softmax(logits[soft_idx] / tau_sharp, dim=1)

        # 5. 计算交叉熵损失
        # 将模型输出的 logits 除以 tau_base 软化后计算 log_softmax
        preds = F.log_softmax(logits / tau_base, dim=1)
        # 计算目标分布与预测分布的交叉熵
        loss = -torch.sum(targets * preds, dim=1).mean()

        return loss

def compute_entropy_loss(logits_over_tau):
        """
        边缘熵最大化正则化 (Marginal Entropy Maximization)
        强制模型在整个 Batch 上对各个类别的预测概率尽量均匀，防止所有样本聚成一类
        """
        probs = F.softmax(logits_over_tau, dim=1)
        mean_probs = probs.mean(dim=0)  # 计算 Batch 内所有样本在各个类别上的平均概率

        # 限制最小值防止 log(0) 导致 NaN
        mean_probs = torch.clamp(mean_probs, min=1e-8)
        loss_entropy = torch.sum(mean_probs * torch.log(mean_probs))

        return loss_entropy

def compute_proto_sep_loss(normed_prototypes, device):
        """
        原型分离正则化 (Prototype Separation)
        显式地推开特征超球面上的各个类别的原型中心，增大类间距离
        """
        tau_sep = 0.1
        num_classes = normed_prototypes.size(0)

        # 计算 K 个原型两两之间的余弦相似度矩阵 (K x K)
        sim_matrix = torch.matmul(normed_prototypes, normed_prototypes.T) / tau_sep

        # 创建一个对角线为 0、其余为 1 的 mask，用于排除原型与自身的相似度
        mask = torch.ones((num_classes, num_classes), device=device) - torch.eye(num_classes, device=device)

        # 目标：最小化不同原型之间的相似度
        exp_sim = torch.exp(sim_matrix) * mask
        sum_exp_sim = exp_sim.sum(dim=1) / (num_classes - 1)
        sum_exp_sim = torch.clamp(sum_exp_sim, min=1e-8)

        loss_sep = torch.log(sum_exp_sim).mean()
        return loss_sep

def _to_bvk(logits_views):
    """
    Convert logits_views to shape [B, V, K].

    Accepts:
    1) Tensor of shape [B, V, K]
    2) list/tuple of V tensors, each [B, K]
    """
    if torch.is_tensor(logits_views):
        if logits_views.dim() != 3:
            raise ValueError(f"logits_views must be [B, V, K], but got {tuple(logits_views.shape)}")
        return logits_views

    if isinstance(logits_views, (list, tuple)):
        if len(logits_views) == 0:
            raise ValueError("logits_views list/tuple is empty.")
        for i, x in enumerate(logits_views):
            if not torch.is_tensor(x):
                raise TypeError(f"logits_views[{i}] is not a Tensor.")
            if x.dim() != 2:
                raise ValueError(f"logits_views[{i}] must be [B, K], but got {tuple(x.shape)}")
        return torch.stack(logits_views, dim=1)

    raise TypeError("logits_views must be a Tensor [B, V, K] or a list/tuple of [B, K] tensors.")


def _default_view_mask(B, V, device):
    return torch.ones(B, V, dtype=torch.bool, device=device)


def _safe_labels(labels, labeled_mask):
    """
    Make sure unlabeled positions do not break indexing.
    Unlabeled positions are replaced with 0 only for safe gather/index ops.
    """
    safe = labels.clone().long()
    safe[~labeled_mask] = 0
    return safe


def _kl_prob_teacher_student(teacher, student, eps=1e-8):
    """
    KL(teacher || student), both are probabilities.

    teacher: [..., K]
    student: [..., K]
    return:  [...]
    """
    teacher = teacher.clamp_min(eps)
    student = student.clamp_min(eps)
    return torch.sum(teacher * (torch.log(teacher) - torch.log(student)), dim=-1)


class MultiViewCategoryCMILoss(nn.Module):
    """
    Multi-view category-level CMI.

    Works on logits/probabilities, not on embeddings.

    For each labeled sample and each valid view, the target is a mixture of:
    1) same-view class centroid
    2) cross-view global class centroid
    3) other-view instance consensus

    This is much more stable for multi-view discovery than using a pure
    single-view class centroid target.

    Args:
        num_classes: total number of output classes K
        topk: number of largest non-GT entries to suppress in refined target
        eps: numerical stability epsilon
        alpha_view: weight for same-view centroid target
        alpha_global: weight for global centroid target
        alpha_instance: weight for other-view consensus target
        confidence_weight: whether to weight loss by target confidence
    """

    def __init__(
        self,
        num_classes: int,
        topk: int = 1,
        eps: float = 1e-8,
        alpha_view: float = 0.6,
        alpha_global: float = 0.3,
        alpha_instance: float = 0.1,
        confidence_weight: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.topk = topk
        self.eps = eps
        self.alpha_view = alpha_view
        self.alpha_global = alpha_global
        self.alpha_instance = alpha_instance
        self.confidence_weight = confidence_weight

    def forward(
        self,
        logits_views,
        labels,
        labeled_mask=None,
        view_mask=None,
        use_refined_target=True,
        return_details=False,
    ):
        """
        Args:
            logits_views:
                [B, V, K] tensor
                or list/tuple of V tensors [B, K]
            labels:
                [B], class labels for labeled samples.
                Unlabeled positions can be any value if labeled_mask=False.
            labeled_mask:
                [B] bool. True means this sample is labeled and participates
                in category-level CMI.
            view_mask:
                [B, V] bool. False means this view is missing/invalid for that sample.
            use_refined_target:
                whether to apply refined target distribution
            return_details:
                whether to return debug info

        Returns:
            loss
            or (loss, details)
        """
        logits_views = _to_bvk(logits_views)  # [B, V, K]
        B, V, K = logits_views.shape
        device = logits_views.device

        if K != self.num_classes:
            raise ValueError(f"num_classes={self.num_classes}, but logits last dim is {K}")

        labels = labels.long().to(device)

        if labeled_mask is None:
            labeled_mask = torch.ones(B, dtype=torch.bool, device=device)
        else:
            labeled_mask = labeled_mask.to(device).bool()

        if view_mask is None:
            view_mask = _default_view_mask(B, V, device)
        else:
            view_mask = view_mask.to(device).bool()

        if labeled_mask.sum() == 0:
            zero = logits_views.sum() * 0.0
            if return_details:
                return zero, {}
            return zero

        probs = F.softmax(logits_views, dim=-1)          # [B, V, K]
        probs_det = probs.detach()

        safe_lbl = _safe_labels(labels, labeled_mask)

        # 1) view-specific centroids
        centroids_view, counts_view = self.compute_view_centroids(
            probs_det, safe_lbl, labeled_mask, view_mask
        )  # [V, K, K], [V, K]

        # 2) global centroids across views
        centroids_global = self.compute_global_centroids(
            centroids_view, counts_view
        )  # [K, K]

        # 3) other-view instance consensus
        other_consensus = self.compute_other_view_consensus(
            probs_det, view_mask
        )  # [B, V, K]

        # Gather per-sample class-conditioned targets
        # q_view:   [B, V, K]
        q_view = centroids_view[:, safe_lbl, :].permute(1, 0, 2)

        # q_global: [B, V, K]
        q_global = centroids_global[safe_lbl].unsqueeze(1).expand(-1, V, -1)

        alpha_sum = self.alpha_view + self.alpha_global + self.alpha_instance
        if alpha_sum <= 0:
            raise ValueError("alpha_view + alpha_global + alpha_instance must be > 0.")

        target = (
            self.alpha_view * q_view
            + self.alpha_global * q_global
            + self.alpha_instance * other_consensus
        ) / alpha_sum

        if use_refined_target:
            target = self.refine_target_distribution(target, safe_lbl)

        target = target.detach()

        # Only labeled and valid views participate
        valid_mask = labeled_mask.unsqueeze(1) & view_mask  # [B, V]

        kl_each = _kl_prob_teacher_student(target, probs, self.eps)  # [B, V]

        if self.confidence_weight:
            weights = target.max(dim=-1).values.detach() * valid_mask.float()
            denom = weights.sum().clamp_min(self.eps)
            loss = (kl_each * weights).sum() / denom
        else:
            denom = valid_mask.float().sum().clamp_min(1.0)
            loss = (kl_each * valid_mask.float()).sum() / denom

        if return_details:
            details = {
                "probs": probs,
                "centroids_view": centroids_view,
                "counts_view": counts_view,
                "centroids_global": centroids_global,
                "other_consensus": other_consensus,
                "target": target,
                "valid_mask": valid_mask,
                "kl_each": kl_each,
            }
            return loss, details

        return loss

    def compute_view_centroids(self, probs, labels, labeled_mask, view_mask):
        """
        probs: [B, V, K]
        labels: [B]
        labeled_mask: [B]
        view_mask: [B, V]

        Returns:
            centroids_view: [V, K, K]
            counts_view:    [V, K]
        """
        B, V, K = probs.shape
        device = probs.device
        dtype = probs.dtype

        centroids = torch.zeros(V, K, K, device=device, dtype=dtype)
        counts = torch.zeros(V, K, device=device, dtype=dtype)
        uniform = torch.full((K,), 1.0 / K, device=device, dtype=dtype)

        for v in range(V):
            for c in range(K):
                mask = labeled_mask & view_mask[:, v] & (labels == c)
                n = int(mask.sum().item())
                if n > 0:
                    pv = probs[mask, v, :]  # [Nc, K]
                    if self.confidence_weight:
                        w = pv.max(dim=-1).values.detach()  # [Nc]
                        centroids[v, c] = (pv * w.unsqueeze(-1)).sum(dim=0) / w.sum().clamp_min(self.eps)
                    else:
                        centroids[v, c] = pv.mean(dim=0)
                    counts[v, c] = float(n)
                else:
                    centroids[v, c] = uniform
                    counts[v, c] = 0.0

        return centroids, counts

    def compute_global_centroids(self, centroids_view, counts_view):
        """
        centroids_view: [V, K, K]
        counts_view: [V, K]
        returns:
            centroids_global: [K, K]
        """
        V, K, _ = centroids_view.shape
        device = centroids_view.device
        dtype = centroids_view.dtype

        centroids_global = torch.zeros(K, K, device=device, dtype=dtype)
        uniform = torch.full((K,), 1.0 / K, device=device, dtype=dtype)

        for c in range(K):
            w = counts_view[:, c]  # [V]
            if w.sum() > 0:
                centroids_global[c] = (centroids_view[:, c, :] * w.unsqueeze(-1)).sum(dim=0) / w.sum()
            else:
                centroids_global[c] = uniform

        return centroids_global

    def compute_other_view_consensus(self, probs, view_mask):
        """
        probs: [B, V, K]
        view_mask: [B, V]

        returns:
            other_consensus: [B, V, K]
        """
        B, V, K = probs.shape
        valid = view_mask.float().unsqueeze(-1)      # [B, V, 1]
        probs_masked = probs * valid                 # [B, V, K]

        sum_all = probs_masked.sum(dim=1, keepdim=True)      # [B, 1, K]
        cnt_all = view_mask.float().sum(dim=1, keepdim=True) # [B, 1]

        sum_others = sum_all - probs_masked                  # [B, V, K]
        cnt_others = cnt_all - view_mask.float()             # [B, V]

        other_consensus = sum_others / cnt_others.clamp_min(1.0).unsqueeze(-1)

        # If a sample has no other valid views, fall back to its own detached probs
        fallback_mask = (cnt_others <= 0).unsqueeze(-1)      # [B, V, 1]
        other_consensus = torch.where(fallback_mask, probs.detach(), other_consensus)

        other_consensus = other_consensus.clamp_min(self.eps)
        other_consensus = other_consensus / other_consensus.sum(dim=-1, keepdim=True)
        return other_consensus

    def refine_target_distribution(self, target, labels):
        """
        target: [B, V, K]
        labels: [B]

        Refine target:
        1) set GT class prob to 1
        2) suppress top-k largest non-GT classes to 0
        3) keep other small non-GT entries
        4) renormalize
        """
        B, V, K = target.shape
        q_hat = target.clone()

        gt_idx = labels.view(B, 1, 1).expand(-1, V, 1)  # [B, V, 1]
        q_hat.scatter_(2, gt_idx, 1.0)

        if self.topk > 0 and K > 1:
            class_indices = torch.arange(K, device=target.device)
            for i in range(B):
                y = int(labels[i].item())
                non_gt_idx = class_indices[class_indices != y]
                if len(non_gt_idx) == 0:
                    continue

                k = min(self.topk, len(non_gt_idx))
                for v in range(V):
                    vals = q_hat[i, v, non_gt_idx]
                    topk_pos = torch.topk(vals, k=k, largest=True).indices
                    topk_idx = non_gt_idx[topk_pos]
                    q_hat[i, v, topk_idx] = 0.0

        q_hat = q_hat.clamp_min(self.eps)
        q_hat = q_hat / q_hat.sum(dim=-1, keepdim=True)
        return q_hat


class MultiViewInstanceCMILoss(nn.Module):
    """
    Multi-view instance-level CMI.

    This is designed for multi-view settings and supports:
    1) anchor_to_others
    2) leave_one_out

    It uses sharpened detached teacher distributions to supervise students.

    Args:
        temperature: sharpen temperature
        eps: numerical stability epsilon
        mode: "anchor_to_others" or "leave_one_out"
        anchor_idx: which view is anchor if mode="anchor_to_others"
        conf_threshold: teacher confidence threshold
        anchor_back_weight: optional weak reverse guidance to anchor
    """

    def __init__(
        self,
        temperature: float = 0.5,
        eps: float = 1e-8,
        mode: str = "anchor_to_others",
        anchor_idx: int = 0,
        conf_threshold: float = 0.0,
        anchor_back_weight: float = 0.1,
    ):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.mode = mode
        self.anchor_idx = anchor_idx
        self.conf_threshold = conf_threshold
        self.anchor_back_weight = anchor_back_weight

    def sharpen(self, p):
        p = p.clamp_min(self.eps)
        p = p ** (1.0 / self.temperature)
        p = p / p.sum(dim=-1, keepdim=True)
        return p

    def forward(self, logits_views, view_mask=None, return_details=False):
        """
        Args:
            logits_views:
                [B, V, K] tensor
                or list/tuple of V tensors [B, K]
            view_mask:
                [B, V] bool. False means this view is missing/invalid.
            return_details:
                whether to return debug info

        Returns:
            loss
            or (loss, details)
        """
        logits_views = _to_bvk(logits_views)  # [B, V, K]
        B, V, K = logits_views.shape
        device = logits_views.device

        if view_mask is None:
            view_mask = _default_view_mask(B, V, device)
        else:
            view_mask = view_mask.to(device).bool()

        if V < 2:
            zero = logits_views.sum() * 0.0
            if return_details:
                return zero, {}
            return zero

        probs = F.softmax(logits_views, dim=-1)   # [B, V, K]
        q = self.sharpen(probs).detach()          # [B, V, K]

        total_num = logits_views.sum() * 0.0
        total_den = logits_views.sum() * 0.0
        details = {
            "probs": probs,
            "teacher_probs": q,
        }

        if self.mode == "anchor_to_others":
            if not (0 <= self.anchor_idx < V):
                raise ValueError(f"anchor_idx={self.anchor_idx} out of range for V={V}")

            teacher = q[:, self.anchor_idx, :]                    # [B, K]
            teacher_conf = teacher.max(dim=-1).values.detach()    # [B]

            for v in range(V):
                if v == self.anchor_idx:
                    continue

                valid = view_mask[:, self.anchor_idx] & view_mask[:, v]
                if self.conf_threshold > 0:
                    valid = valid & (teacher_conf >= self.conf_threshold)

                if valid.any():
                    kl = _kl_prob_teacher_student(
                        teacher[valid], probs[valid, v, :], self.eps
                    )  # [Nv]
                    w = teacher_conf[valid]
                    total_num = total_num + (kl * w).sum()
                    total_den = total_den + w.sum()
                    if return_details:
                        details[f"anchor_to_view{v}_mean_kl"] = kl.mean().detach()

            # Optional weak reverse guidance: other valid views -> anchor
            if self.anchor_back_weight > 0:
                other_mask = view_mask.clone()
                other_mask[:, self.anchor_idx] = False

                other_count = other_mask.float().sum(dim=1)  # [B]
                valid = view_mask[:, self.anchor_idx] & (other_count > 0)

                if valid.any():
                    teacher_anchor = self.compute_other_teacher_for_anchor(q, other_mask)
                    teacher_anchor_conf = teacher_anchor.max(dim=-1).values.detach()

                    if self.conf_threshold > 0:
                        valid = valid & (teacher_anchor_conf >= self.conf_threshold)

                    if valid.any():
                        kl = _kl_prob_teacher_student(
                            teacher_anchor[valid],
                            probs[valid, self.anchor_idx, :],
                            self.eps,
                        )
                        w = teacher_anchor_conf[valid]
                        total_num = total_num + self.anchor_back_weight * (kl * w).sum()
                        total_den = total_den + self.anchor_back_weight * w.sum()
                        if return_details:
                            details["others_to_anchor_mean_kl"] = kl.mean().detach()

        elif self.mode == "leave_one_out":
            for v in range(V):
                other_mask = view_mask.clone()
                other_mask[:, v] = False
                other_count = other_mask.float().sum(dim=1)  # [B]
                valid = view_mask[:, v] & (other_count > 0)

                if valid.any():
                    teacher = self.compute_leave_one_out_teacher(q, other_mask)
                    teacher_conf = teacher.max(dim=-1).values.detach()

                    if self.conf_threshold > 0:
                        valid = valid & (teacher_conf >= self.conf_threshold)

                    if valid.any():
                        kl = _kl_prob_teacher_student(
                            teacher[valid], probs[valid, v, :], self.eps
                        )
                        w = teacher_conf[valid]
                        total_num = total_num + (kl * w).sum()
                        total_den = total_den + w.sum()
                        if return_details:
                            details[f"loo_to_view{v}_mean_kl"] = kl.mean().detach()
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Choose from ['anchor_to_others', 'leave_one_out'].")

        if total_den.item() == 0:
            loss = logits_views.sum() * 0.0
        else:
            loss = total_num / total_den.clamp_min(self.eps)

        if return_details:
            return loss, details
        return loss

    def compute_other_teacher_for_anchor(self, q, other_mask):
        """
        q: [B, V, K]
        other_mask: [B, V]
        returns:
            teacher_anchor: [B, K]
        """
        valid = other_mask.float().unsqueeze(-1)  # [B, V, 1]
        q_masked = q * valid                      # [B, V, K]
        sum_q = q_masked.sum(dim=1)               # [B, K]
        count = other_mask.float().sum(dim=1).clamp_min(1.0).unsqueeze(-1)  # [B, 1]
        teacher = sum_q / count
        teacher = teacher.clamp_min(self.eps)
        teacher = teacher / teacher.sum(dim=-1, keepdim=True)
        return teacher

    def compute_leave_one_out_teacher(self, q, other_mask):
        """
        q: [B, V, K]
        other_mask: [B, V]
        returns:
            teacher: [B, K]
        """
        valid = other_mask.float().unsqueeze(-1)
        q_masked = q * valid
        sum_q = q_masked.sum(dim=1)  # [B, K]
        count = other_mask.float().sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        teacher = sum_q / count
        teacher = teacher.clamp_min(self.eps)
        teacher = teacher / teacher.sum(dim=-1, keepdim=True)
        return teacher


def multi_view_entropy_balance_loss(logits_views, view_mask=None, eps=1e-8):
    """
    Negative entropy on marginal predictions:
    minimizing this encourages balanced cluster usage.

    Returns:
        sum_k p_bar(k) * log p_bar(k)
    """
    logits_views = _to_bvk(logits_views)  # [B, V, K]
    B, V, K = logits_views.shape
    device = logits_views.device

    probs = F.softmax(logits_views, dim=-1)  # [B, V, K]

    if view_mask is None:
        p_bar = probs.mean(dim=(0, 1))  # [K]
    else:
        view_mask = view_mask.to(device).bool()
        valid = view_mask.float().unsqueeze(-1)  # [B, V, 1]
        denom = valid.sum().clamp_min(1.0)
        p_bar = (probs * valid).sum(dim=(0, 1)) / denom

    p_bar = p_bar.clamp_min(eps)
    p_bar = p_bar / p_bar.sum()
    return torch.sum(p_bar * torch.log(p_bar))


class MultiViewCMILossBundle(nn.Module):
    """
    Convenient bundle:
    total = lambda_cat * category_cmi + lambda_inst * instance_cmi + lambda_ent * entropy_balance

    This only computes the CMI-related auxiliary losses.
    You can add it to your main task loss outside.
    """

    def __init__(
        self,
        num_classes: int,
        lambda_cat: float = 0.05,
        lambda_inst: float = 0.10,
        lambda_ent: float = 1.0,
        category_kwargs=None,
        instance_kwargs=None,
    ):
        super().__init__()
        category_kwargs = {} if category_kwargs is None else category_kwargs
        instance_kwargs = {} if instance_kwargs is None else instance_kwargs

        self.lambda_cat = lambda_cat
        self.lambda_inst = lambda_inst
        self.lambda_ent = lambda_ent

        self.category_loss = MultiViewCategoryCMILoss(
            num_classes=num_classes,
            **category_kwargs,
        )
        self.instance_loss = MultiViewInstanceCMILoss(
            **instance_kwargs,
        )

    def forward(
        self,
        logits_views,
        labels=None,
        labeled_mask=None,
        view_mask=None,
        return_details=False,
    ):
        total = 0.0
        details = {}

        # Category-level loss
        if self.lambda_cat > 0 and labels is not None:
            loss_cat, det_cat = self.category_loss(
                logits_views,
                labels,
                labeled_mask=labeled_mask,
                view_mask=view_mask,
                return_details=True,
            )
            total = total + self.lambda_cat * loss_cat
            details["loss_cat"] = loss_cat.detach()
            details["cat_details"] = det_cat
        else:
            loss_cat = None

        # Instance-level loss
        if self.lambda_inst > 0:
            loss_inst, det_inst = self.instance_loss(
                logits_views,
                view_mask=view_mask,
                return_details=True,
            )
            total = total + self.lambda_inst * loss_inst
            details["loss_inst"] = loss_inst.detach()
            details["inst_details"] = det_inst
        else:
            loss_inst = None

        # Entropy balance
        if self.lambda_ent > 0:
            loss_ent = multi_view_entropy_balance_loss(
                logits_views,
                view_mask=view_mask,
            )
            total = total + self.lambda_ent * loss_ent
            details["loss_ent"] = loss_ent.detach()
        else:
            loss_ent = None

        if return_details:
            details["loss_total_aux"] = total.detach() if torch.is_tensor(total) else total
            return total, details
        return total


