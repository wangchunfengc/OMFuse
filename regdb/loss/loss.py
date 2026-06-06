import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable


import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphStripeRelationalLoss(nn.Module):
    """
    Graph-Stripe Relational Loss (GSR-Loss)
    输入:
        feat: 形状 [B, S*d] 或 [B, S, d] 的拼接局部特征（例如 final_feat_all 或 feat_all）
        labels: [B] 的行向量，整数ID
    超参:
        num_stripes: S, 条带数 (与你模型里的 num_stripes 一致)
        local_dim:   d, 每条带的维度 (与你模型里的 local_feat_dim 一致)
        margin:      triplet 的 margin
        lambda_graph:图一致性损失权重
        temperature: 构建条带图时的 softmax 温度（越小越“尖锐”）
        normalize:   是否对局部向量做 L2 normalize
    """
    def __init__(
        self,
        num_stripes: int = 6,
        local_dim: int = 256,
        margin: float = 0.3,
        lambda_graph: float = 0.2,
        temperature: float = 0.07,
        normalize: bool = True,
    ):
        super().__init__()
        self.S = num_stripes
        self.d = local_dim
        self.margin = margin
        self.lambda_graph = lambda_graph
        self.tau = temperature
        self.normalize = normalize
        self.ranking = nn.MarginRankingLoss(margin=margin)

    def _reshape(self, feat: torch.Tensor) -> torch.Tensor:
        """把 [B, S*d] 或 [B, S, d] 规整成 [B, S, d]"""
        if feat.dim() == 2:
            B, Sd = feat.shape
            assert Sd % self.S == 0, f"feat维度不匹配: {Sd} 不是 num_stripes={self.S} 的整数倍"
            d = Sd // self.S
            x = feat.view(B, self.S, d)
        elif feat.dim() == 3:
            B, S, d = feat.shape
            assert S == self.S, f"传入条带数 S={S} 与设置的 num_stripes={self.S} 不一致"
            x = feat
        else:
            raise ValueError("feat 形状必须是 [B, S*d] 或 [B, S, d]")
        if self.normalize:
            x = F.normalize(x, dim=2)
        return x  # [B, S, d]

    @staticmethod
    def _batch_hard_triplet(x: torch.Tensor, labels: torch.Tensor, margin: float):
        """
        在维度 [B, d] 上做 batch-hard triplet，返回均值损失。
        x 已经 L2 normalize。
        """
        B = x.size(0)
        # 余弦距离 -> 欧氏等价：dist = 1 - cos
        sim = x @ x.t()                     # [B, B]
        dist = (1. - sim).clamp(min=0.)     # [B, B]

        labels = labels.view(-1, 1)
        mask_pos = labels.eq(labels.t())    # 同ID
        mask_neg = ~mask_pos

        # 对角线不算正样本
        mask_pos = mask_pos.fill_diagonal_(False)

        # hardest positive: 同ID里最大距离
        pos_dist = dist.clone()
        pos_dist[~mask_pos] = -1e9
        hardest_pos, _ = pos_dist.max(dim=1)  # [B]

        # hardest negative: 不同ID里最小距离
        neg_dist = dist.clone()
        neg_dist[~mask_neg] = 1e9
        hardest_neg, _ = neg_dist.min(dim=1)  # [B]

        # 仅对有正样本的 anchor 计损
        valid = (hardest_pos > -1e8)
        if valid.any():
            y = torch.ones_like(hardest_pos[valid])
            loss = F.margin_ranking_loss(
                hardest_neg[valid],
                hardest_pos[valid],
                y,
                margin=margin,
                reduction='mean'
            )
        else:
            loss = dist.new_tensor(0.0)

        return loss

    def _stripe_graph(self, x_stripes: torch.Tensor) -> torch.Tensor:
        """
        构建每个样本的条带图：
        输入: x_stripes [B, S, d]（已归一化）
        输出: A [B, S, S],  A_b = softmax( (x_b x x_b^T)/tau ) 沿最后一维
        """
        B, S, d = x_stripes.shape
        # 条带间相似度：每个样本独立计算
        sim = torch.bmm(x_stripes, x_stripes.transpose(1, 2))  # [B, S, S]
        A = F.softmax(sim / self.tau, dim=-1)                  # 行归一化
        return A

    def _graph_consistency(self, x_stripes: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        同ID样本的条带图一致性（MSE）。
        """
        A = self._stripe_graph(x_stripes)  # [B, S, S]
        B = A.size(0)
        labels = labels.view(-1, 1)
        mask_pos = labels.eq(labels.t()).fill_diagonal_(False)  # [B, B]

        if mask_pos.sum() == 0:
            return A.new_tensor(0.0)

        # 计算所有正对 (i,j) 的 ||A_i - A_j||_F^2
        # 向量化做法：扩一维再相减
        A_i = A.unsqueeze(1).expand(B, B, self.S, self.S)  # [B, B, S, S]
        A_j = A.unsqueeze(0).expand(B, B, self.S, self.S)  # [B, B, S, S]
        diff = (A_i - A_j).pow(2).sum(dim=(-1, -2))        # [B, B]
        loss = diff[mask_pos].mean()
        return loss

    def forward(self, feat: torch.Tensor, labels: torch.Tensor):
        """
        返回: 总损失, 以及字典{'triplet':..., 'graph': ...}
        """
        x = self._reshape(feat)          # [B, S, d]
        B, S, d = x.shape

        # 逐条带 Triplet（batch-hard）
        loss_trip = 0.0
        for s in range(S):
            loss_trip = loss_trip + self._batch_hard_triplet(x[:, s, :], labels, self.margin)
        loss_trip = loss_trip / S

        # 条带图一致性
        loss_graph = self._graph_consistency(x, labels)

        loss = loss_trip + self.lambda_graph * loss_graph
        return loss, {'triplet': loss_trip.detach(), 'graph': loss_graph.detach()}

        
def comp_dist(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    #torch.Size([96, 1024])
    m, n = emb1.shape[0], emb2.shape[0]
    #a2+b2
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    #dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    #(a-b)2
    dist_mtx = dist_mtx.addmm_(emb1, emb2.t(), beta=1, alpha=-2)
    
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx   


import torch
import torch.nn as nn
import torch.nn.functional as F

class CenterAggregationLossWithGraph(nn.Module):
    """
    基于原 CAL，融入图思想：
      - base: 原来的跨模态中心配对项（做了归一化，数值更稳）
      - contrast: 跨模态 InfoNCE（RGB中心 ↔ IR中心，对齐同ID，其他为负）
      - geom: 两模态中心的几何一致性（pairwise 相似度矩阵对齐，off-diagonal）
    输入:
      inputs: (B, D)，按 [RGB前半, IR后半] 拼接
      targets: (B,)
    超参:
      temperature: InfoNCE 温度
      lambda_con: 对比项权重
      lambda_geom: 几何一致性权重
      k_hard: 若想只用 top-k 困难负样本，可改造对比项；当前使用“全负样本”更稳定
      normalize: 是否先 L2 归一化
    """
    def __init__(self, margin=0.1, temperature=0.07,
                 lambda_con=0.2, lambda_geom=0.1,
                 normalize=True):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
        self.lambda_con = lambda_con
        self.lambda_geom = lambda_geom
        self.normalize = normalize

    @staticmethod
    def _cosine_dist(a, b):
        # a: (N, D), b: (M, D)  ->  (N, M)  距离 = 1 - cos
        return 1.0 - a @ b.t()

    def forward(self, inputs, targets):
        B, D = inputs.size()
        assert B % 2 == 0, "batch 必须是 RGB/IR 各一半拼接"
        m = B // 2

        x_rgb, y_rgb = inputs[:m], targets[:m]
        x_ir , y_ir  = inputs[m:], targets[m:]

        # 归一化更稳（对比/几何都基于 cos）
        if self.normalize:
            x_rgb = F.normalize(x_rgb, dim=1)
            x_ir  = F.normalize(x_ir , dim=1)

        # ---------- (1) 原中心配对项：同ID的 RGB/IR 中心距离 ----------
        # 按你原来的写法：对每个样本 i，都取它所属 ID 的“批内中心”
        centers_rgb = torch.stack([x_rgb[y_rgb == y_rgb[i]].mean(0) for i in range(m)], dim=0)  # (m,D)
        centers_ir  = torch.stack([x_ir [y_ir  == y_ir [i]].mean(0) for i in range(m)], dim=0)  # (m,D)

        # 同索引视为同ID（因中心由 y==y[i] 求得）
        dist_pos = 1.0 - torch.sum(centers_rgb * centers_ir, dim=1)  # (m,)  = cos 距离

        # 为了和你原始风格靠近，这里给一个稳定的 “正/负 比率” 形式（负样本=跨模态其他ID距离的均值）
        cross_dist = self._cosine_dist(centers_rgb, centers_ir)      # (m,m)
        pos_mask   = (y_rgb.view(-1,1) == y_ir.view(1,-1))           # (m,m)
        neg_mask   = ~pos_mask

        # 均值负距离（排除正对角）
        neg_mean_rgb = (cross_dist * neg_mask.float()).sum(dim=1) / (neg_mask.float().sum(dim=1) + 1e-6)  # (m,)
        neg_mean_ir  = (cross_dist.t() * neg_mask.t().float()).sum(dim=1) / (neg_mask.t().float().sum(dim=1) + 1e-6)  # (m,)

        base_loss = dist_pos.mean() / ( (neg_mean_rgb.mean() + neg_mean_ir.mean() + 1e-6) )

        # ---------- (2) 图对比（InfoNCE）：以“ID中心”为节点进行跨模态对比 ----------
        # 统一到“每个 ID 一个中心”，避免同 ID 重复计权
        ids_rgb = y_rgb.unique()
        ids_ir  = y_ir.unique()
        common_ids = torch.tensor(sorted(list(set(ids_rgb.tolist()) & set(ids_ir.tolist()))),
                                  device=inputs.device, dtype=y_rgb.dtype)
        # 如果 batch 采样保证 RGB/IR 都含有相同 ID，这里一般 M>0
        if common_ids.numel() > 0:
            c_rgb = torch.stack([x_rgb[y_rgb == i].mean(0) for i in common_ids], dim=0)  # (M,D)
            c_ir  = torch.stack([x_ir [y_ir  == i].mean(0) for i in common_ids], dim=0)  # (M,D)
            if self.normalize:
                c_rgb = F.normalize(c_rgb, dim=1)
                c_ir  = F.normalize(c_ir , dim=1)

            logits = (c_rgb @ c_ir.t()) / self.temperature                 # (M,M)
            targets_con = torch.arange(logits.size(0), device=inputs.device)
            loss_con = 0.5 * (F.cross_entropy(logits, targets_con) +
                               F.cross_entropy(logits.t(), targets_con))
        else:
            loss_con = inputs.new_tensor(0.0)

        # ---------- (3) 几何一致性：两模态中心的相对结构对齐 ----------
        # 用 pairwise 相似度（cos），只约束 off-diagonal，避免塌缩到对角
        if common_ids.numel() >= 2:
            S_rgb = c_rgb @ c_rgb.t()  # (M,M)
            S_ir  = c_ir  @ c_ir.t()   # (M,M)
            off = ~torch.eye(S_rgb.size(0), dtype=torch.bool, device=inputs.device)
            loss_geom = F.mse_loss(S_rgb[off], S_ir[off])
        else:
            loss_geom = inputs.new_tensor(0.0)

        loss = base_loss + self.lambda_con * loss_con + self.lambda_geom * loss_geom
        return loss


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class IntraModalCompactLoss(nn.Module):
    """
    仅约束“模态内”：
      L = L_proto_nce + λ_var * L_var + λ_sep * L_center_sep + λ_local * L_local
    - L_proto_nce: anchor(样本) vs 同模态各ID原型 的 SupCon(CE)；正类为自身ID原型（排除自身后的proto）
    - L_var: 样本到自身ID原型(同模态)的 SmoothL1 方差收缩（proto上 stop-grad）
    - L_center_sep: 同模态不同ID原型若 cos > m_sep 则 hinge 惩罚
    - L_local: 局部分支到各自同模态ID原型的 SmoothL1（proto上 stop-grad）

    约定 batch 前半 VIS、后半 IR。
    """
    def __init__(self, temp=0.07, lambda_var=0.1, lambda_sep=0.02, lambda_local=0.1,
                 m_sep=0.1, normalize=True, eps=1e-6, topk_sep=None):
        super().__init__()
        self.temp = temp
        self.lambda_var = lambda_var
        self.lambda_sep = lambda_sep
        self.lambda_local = lambda_local
        self.m_sep = m_sep
        self.normalize = normalize
        self.eps = eps
        self.topk_sep = topk_sep  # 若不为None，仅对最相近的 top-k 原型对做分离惩罚

    @staticmethod
    def _proto_stats(feat, label):
        """
        给定同一模态的 (n, C) 特征与标签，返回：
          proto: (K, C) 原型向量（每个ID一个）
          ids:   (K,)   原型对应的ID
          sum_g, cnt_g: 每个ID的向量和与个数（用于“去自身”原型）
        """
        ids, inv = torch.unique_consecutive(label.sort()[0], return_inverse=True)  # 排序稳定
        # 用 scatter 统计
        C = feat.size(1)
        sum_g = torch.zeros(ids.numel(), C, device=feat.device, dtype=feat.dtype)   #统计id相同的feat
        cnt_g = torch.zeros(ids.numel(), 1, device=feat.device, dtype=feat.dtype)       #统计id相同的数量
        sum_g.index_add_(0, inv, feat[label.argsort()])
        cnt_g.index_add_(0, inv, torch.ones_like(cnt_g).expand(-1,1)[inv])
        proto = sum_g / (cnt_g + 1e-6)
        return proto, ids, sum_g, cnt_g

    def _proto_for_each_sample_excluding_self(self, feat, label, sum_g, cnt_g, ids):
        """
        为每个样本构造“排除自身”的同类原型：
          proto_excl[i] = (sum_yi - x_i) / (cnt_yi - 1)
        若该ID仅出现一次，则标记为无正样本（mask=False）。
        """
        # 建立 id -> 索引
        id2idx = {int(k.item()): i for i, k in enumerate(ids)}
        idxs = torch.tensor([id2idx[int(y.item())] for y in label], device=feat.device)
        sum_y = sum_g[idxs]   # (n, C)
        cnt_y = cnt_g[idxs]   # (n, 1)
        proto_excl = (sum_y - feat) / torch.clamp(cnt_y - 1.0, min=1.0)  # 分母<1的地方会被mask掉
        has_pos = (cnt_y.view(-1) >= 2.0)  # 只有出现≥2次才有正样本
        return proto_excl, has_pos

    def _proto_supcon_ce(self, feat, label):
        if feat.size(0) <= 1:
            return feat.new_tensor(0.0)

        # 统计同模态原型
        proto, ids, sum_g, cnt_g = self._proto_stats(feat, label)

        if self.normalize:
            feat = F.normalize(feat, dim=1)
            proto = F.normalize(proto, dim=1)

        # “去自身”的同类原型（每个样本一个）
        proto_excl, has_pos = self._proto_for_each_sample_excluding_self(
            feat, label, sum_g, cnt_g, ids
        )
        if has_pos.sum() == 0:
            return feat.new_tensor(0.0)

        # 构造目标索引（每个样本的ID在 proto 中的位置）
        id2idx = {int(k.item()): i for i, k in enumerate(ids)}
        target_idx = torch.tensor([id2idx[int(y.item())] for y in label], device=feat.device)

        # logits = x vs 全部原型（原型可 stop-grad 更稳）
        with torch.no_grad():
            P = proto  # 原型上停止梯度更稳，也可不加 no_grad
        logits = (feat @ P.t()) / self.temp

        # 用“去自身原型”的相似度替换**每个样本**自己的正类 logit（逐样本覆盖，不改整行）
        rows = torch.arange(feat.size(0), device=feat.device)
        pos_logit = (feat * proto_excl).sum(dim=1) / self.temp
        logits[rows[has_pos], target_idx[has_pos]] = pos_logit[has_pos]

        return F.cross_entropy(logits[has_pos], target_idx[has_pos])


    def forward(self, feats, labels, parts=None):
        # 强制在 FP32 上算，数值更稳
        feats = feats.float()
        B = feats.size(0); assert B % 2 == 0, "batch 必须前半VIS后半IR"
        b2 = B // 2

        f_vis, y_vis = feats[:b2], labels[:b2]
        f_ir , y_ir  = feats[b2:], labels[b2:]

        # (1) 原型SupCon（同模态）
        L_nce_vis = self._proto_supcon_ce(f_vis, y_vis)
        L_nce_ir  = self._proto_supcon_ce(f_ir , y_ir)
        L_proto_nce = 0.5 * (L_nce_vis + L_nce_ir)

        #(4) 局部分支到原型（stop-grad）
        L_local = feats.new_tensor(0.0)
        if parts and self.lambda_local > 0:
            acc, n = feats.new_tensor(0.0), 0
            for p in parts:
                p = p.float()
                pv, pi = p[:b2], p[b2:]
                acc += 0.5 * (self._proto_supcon_ce(pv, y_vis) + self._proto_supcon_ce(pi, y_ir))
                n += 1
            if n > 0: L_local = acc / n

        loss = L_proto_nce  +  L_local
        stat = {
            'proto_nce': L_proto_nce.detach(),
            'local': L_local.detach()
        }
        return loss, stat




class CenterAggregationLoss(nn.Module):
    def __init__(self, k_size=4, margin=0.1):
        super(CenterAggregationLoss, self).__init__()
        self.margin = margin
        self.k_size = k_size
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        
        inputsRGB=inputs[0:n//2]
        targetRGB=targets[0:n//2]
        inputsIR=inputs[n//2:n]       
        targetIR=targets[n//2:n] 
        # Come to centers
        centersRGB = []
        centersIR = []
        
        for i in range(n//2):
            centersRGB.append(inputsRGB[targetRGB == targetRGB[i]].mean(0))
            centersIR.append(inputsIR[targetIR == targetIR[i]].mean(0))
        
        #array
        centersRGB = torch.stack(centersRGB)
        centersIR = torch.stack(centersIR)
        # centers:torch.Size([96, 2048]) input：torch.Size([96, 2048])       
        dist_pc = (centersRGB - centersIR)**2
        dist_pc = dist_pc.sum(1)
        dist_pc = dist_pc.sqrt()

        centersRGB=torch.cat([centersRGB,centersRGB])
        centersIR=torch.cat([centersIR,centersIR])
        distRGB = comp_dist(centersRGB,inputs)
        distIR = comp_dist(centersIR,inputs)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_R, dist_I = [], []
        dist_RP, dist_IP = [], []
        beta=0.1
        gama=0.1
        for i in range(0, n):
            dist_R.append(distRGB[i][mask[i] == 0].clamp(min=0.0).mean())
            dist_I.append(distIR[i][mask[i] == 0].clamp(min=0.0).mean())
            dist_RP.append(distRGB[i][mask[i]].clamp(min=0.0).min())
            dist_IP.append(distIR[i][mask[i]].clamp(min=0.0).min())
        dist_R = torch.stack(dist_R)
        dist_I = torch.stack(dist_I)
        dist_RP = torch.stack(dist_RP)
        dist_IP = torch.stack(dist_IP)
        alpha=0
        loss=dist_pc.sum()/((dist_R.sum()+dist_I.sum()-dist_pc.sum()))
        
        return loss #, dist_pc.mean(), dist_an.mean()class OriTripletLoss(nn.Module):  


class OriTripletLoss(nn.Module): 
    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):

        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(),beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss#, correct       
        
# Adaptive weights
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)


        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct
        
def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx    

def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx

class DCL(nn.Module):
    def __init__(self, num_pos=4, feat_norm='no'):
        super(DCL, self).__init__()
        self.num_pos = num_pos
        self.feat_norm = feat_norm

    def forward(self,inputs, targets):
        if self.feat_norm == 'yes':
            inputs = F.normalize(inputs, p=2, dim=-1)
        temps=2
        N = inputs.size(0)
        id_num = N // temps // self.num_pos

        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t())
        is_neg_c2i = is_neg[::self.num_pos, :].chunk(temps, 0)[0]  # mask [id_num, N]

        centers = []
        for i in range(id_num):
            centers.append(inputs[targets == targets[i * self.num_pos]].mean(0))
        centers = torch.stack(centers)

        dist_mat = pdist_torch(centers, inputs)  #  c-i

        an = dist_mat * is_neg_c2i
        an = an[an > 1e-6].view(id_num, -1)

        d_neg = torch.mean(an, dim=1, keepdim=True)
        mask_an = (an - d_neg).expand(id_num, N - temps * self.num_pos).lt(0)  # mask
        an = an * mask_an

        list_an = []
        for i in range (id_num):
            list_an.append(torch.mean(an[i][an[i]>1e-6]))
        an_mean = sum(list_an) / len(list_an)
        #~线的意思就是翻过来true<——>false
        ap = dist_mat * ~is_neg_c2i
        ap_mean = torch.mean(ap[ap>1e-6])

        loss = ap_mean / an_mean

        return loss


class CenterTripletLoss(nn.Module):
    def __init__(self, k_size, margin=0):
        super(CenterTripletLoss, self).__init__()
        self.margin = margin
        self.k_size = k_size
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)

        # Come to centers
        centers = []
        for i in range(n):
            centers.append(inputs[targets == targets[i]].mean(0))
        #array
        centers = torch.stack(centers)
        # centers:torch.Size([96, 2048]) input：torch.Size([96, 2048])       
        dist_pc = (inputs - centers)**2
        dist_pc = dist_pc.sum(1)
        dist_pc = dist_pc.sqrt()

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, centers, centers.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_an, dist_ap = [], []
        for i in range(0, n, self.k_size):
            dist_an.append((self.margin - dist[i][mask[i] == 0]).clamp(min=0.0).mean() )
        dist_an = torch.stack(dist_an)

        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        loss = dist_pc.mean() + dist_an.mean()
        return loss/2#, dist_pc.mean(), dist_an.mean()

       
def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx

class MSEL(nn.Module):
    def __init__(self,num_pos=4,feat_norm = 'no'):
        super(MSEL, self).__init__()
        self.num_pos = num_pos
        self.feat_norm = feat_norm

    def forward(self, inputs, targets):
        if self.feat_norm == 'yes':
            inputs = F.normalize(inputs, p=2, dim=-1)

        target, _ = targets.chunk(2,0)
        N = target.size(0)

        dist_mat = pdist_torch(inputs, inputs)

        dist_intra_rgb = dist_mat[0 : N, 0 : N]
        dist_cross_rgb = dist_mat[0 : N, N : 2*N]
        dist_intra_ir = dist_mat[N : 2*N, N : 2*N]
        dist_cross_ir = dist_mat[N : 2*N, 0 : N]

        # shape [N, N]
        is_pos = target.expand(N, N).eq(target.expand(N, N).t())

        dist_intra_rgb = is_pos * dist_intra_rgb
        #torch.topk(tensor1, k=3, dim=1, largest=True)把tenser中的那个最大的k个拿出来
        intra_rgb, _ = dist_intra_rgb.topk(self.num_pos - 1, dim=1 ,largest = True, sorted = False) # remove itself
        intra_mean_rgb = torch.mean(intra_rgb, dim=1)

        dist_intra_ir = is_pos * dist_intra_ir
        intra_ir, _ = dist_intra_ir.topk(self.num_pos - 1, dim=1, largest=True, sorted=False)
        intra_mean_ir = torch.mean(intra_ir, dim=1)

        dist_cross_rgb = dist_cross_rgb[is_pos].contiguous().view(N, -1)  # [N, num_pos]
        cross_mean_rgb = torch.mean(dist_cross_rgb, dim =1)

        dist_cross_ir = dist_cross_ir[is_pos].contiguous().view(N, -1)  # [N, num_pos]
        cross_mean_ir = torch.mean(dist_cross_ir, dim=1)

        loss = (torch.mean(torch.pow(cross_mean_rgb - intra_mean_rgb, 2)) +
                torch.mean(torch.pow(cross_mean_ir - intra_mean_ir, 2))) / 2

        return loss

class CenterLoss(nn.Module):
    def __init__(self, k_size, margin=0.1):
        super(CenterLoss, self).__init__()
        self.margin = margin
        self.k_size = k_size
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)

        # Come to centers
        centers = []
        for i in range(n):
            centers.append(inputs[targets == targets[i]].mean(0))
        #array
        centers = torch.stack(centers)
        # centers:torch.Size([96, 2048]) input：torch.Size([96, 2048])       
        dist_pc = (inputs - centers)**2
        dist_pc = dist_pc.sum(1)
        dist_pc = dist_pc.sqrt()

        return dist_pc.mean()/2#, dist_pc.mean(), dist_an.mean()

