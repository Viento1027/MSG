import random
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage.segmentation import slic


# ---------- Utils: Superpixel & KNN ----------

def slic_superpixels(x_chw: torch.Tensor,
                     n_segments: int = 10,
                     compactness: float = 10.0,
                     sigma: float = 0.0) -> Tuple[np.ndarray, int]:
    """Compute SLIC superpixels on (C,H,W) -> return (H,W) labels and count."""
    x_np = x_chw.detach().cpu().numpy()
    x_hwc = np.transpose(x_np, (1, 2, 0))
    seg = slic(x_hwc, n_segments=n_segments, compactness=compactness,
               sigma=sigma, start_label=0, channel_axis=-1).astype(np.int32)
    n_sp = int(seg.max()) + 1
    return seg, n_sp


def superpixel_features_and_positions(x_chw: torch.Tensor,
                                      seg: np.ndarray,
                                      n_sp: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aggregate per-superpixel mean spectrum and (row,col) center."""
    C, H, W = x_chw.shape
    X_sp = torch.zeros((n_sp, C), dtype=torch.float32, device=x_chw.device)
    P_sp = torch.zeros((n_sp, 2), dtype=torch.float32, device=x_chw.device)
    counts = torch.zeros((n_sp,), dtype=torch.float32, device=x_chw.device)
    for r in range(H):
        for c in range(W):
            sid = int(seg[r, c])
            X_sp[sid] += x_chw[:, r, c]
            P_sp[sid] += torch.tensor([r, c], dtype=torch.float32, device=x_chw.device)
            counts[sid] += 1.0
    counts = counts.clamp_min(1.0).unsqueeze(-1)
    X_sp = X_sp / counts
    P_sp = P_sp / counts
    return X_sp, P_sp


def pairwise_distances(A: torch.Tensor, B: torch.Tensor, metric: str = "euclidean") -> torch.Tensor:
    """Pairwise distance matrix between rows of A and B."""
    if metric == "euclidean":
        a2 = (A ** 2).sum(dim=1, keepdim=True)
        b2 = (B ** 2).sum(dim=1, keepdim=True).t()
        dist2 = a2 + b2 - 2.0 * (A @ B.t())
        return dist2.clamp_min(0.0).sqrt()
    elif metric == "cosine":
        A = F.normalize(A, dim=-1)
        B = F.normalize(B, dim=-1)
        return 1.0 - (A @ B.t())
    else:
        raise ValueError(f"Unsupported metric: {metric}")


def build_knn_adjacency(X_sp: torch.Tensor,
                        P_sp: torch.Tensor,
                        k: int = 10,
                        eta_spatial: float = 0.5,
                        spectral_metric: str = "euclidean",
                        spatial_metric: str = "euclidean",
                        weight_mode: str = "heat",
                        heat_delta: float = 1.0) -> torch.Tensor:
    """Build a spectral–spatial KNN graph with weighted edges."""
    N = X_sp.shape[0]
    D_spec = pairwise_distances(X_sp, X_sp, metric=spectral_metric)
    D_spat = pairwise_distances(P_sp, P_sp, metric=spatial_metric)

    def _norm(M):
        m = M.max()
        return M / (m + 1e-8) if m > 0 else M

    D = eta_spatial * _norm(D_spat) + (1.0 - eta_spatial) * _norm(D_spec)

    A = torch.zeros((N, N), dtype=torch.float32, device=X_sp.device)
    k_eff = min(k, N - 1) if N > 1 else 0
    if k_eff > 0:
        for i in range(N):
            d_i = D[i].clone()
            d_i[i] = float('inf')
            nn_idx = torch.topk(-d_i, k=k_eff).indices
            for j in nn_idx.tolist():
                A[i, j] = 1.0
                A[j, i] = 1.0

    if weight_mode == "heat":
        W = torch.exp(-(D ** 2) / (heat_delta ** 2 + 1e-8))
    elif weight_mode in ("euclidean", "cosine"):
        W = (1.0 - D).clamp_min(0.0)
    elif weight_mode == "binary":
        W = (A > 0).float()
    else:
        raise ValueError(f"Unsupported weight_mode: {weight_mode}")

    A_weighted = torch.zeros_like(W)
    A_weighted[A > 0] = W[A > 0]
    A_weighted += torch.eye(N, device=X_sp.device)
    return torch.maximum(A_weighted, A_weighted.t())


def k_hop_mask(A_bin: torch.Tensor, center: int, K: int) -> torch.Tensor:
    """Return boolean K-hop neighborhood (incl. center) on a binary adjacency."""
    N = A_bin.size(0)
    if N == 0:
        return torch.zeros(0, dtype=torch.bool, device=A_bin.device)
    A = (A_bin > 0).to(torch.float32)
    visited = torch.zeros(N, dtype=torch.bool, device=A.device)
    frontier = torch.zeros(N, dtype=torch.bool, device=A.device)
    visited[center] = True
    frontier[center] = True
    for _ in range(max(0, K)):
        nxt = (A @ frontier.to(torch.float32)).gt(0)
        frontier = nxt & (~visited)
        visited = visited | frontier
        if not frontier.any():
            break
    return visited


# ---------- Encoder & Projector ----------

class GCNLayer(nn.Module):
    """Single GCN layer with BN + PReLU."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)
        self.act = nn.PReLU()

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        deg = A.sum(dim=-1).clamp_min(1.0)
        d_inv_sqrt = torch.pow(deg, -0.5)
        A_hat = d_inv_sqrt.unsqueeze(0) * A * d_inv_sqrt.unsqueeze(1)
        H = self.lin(X)
        H = A_hat @ H
        H = self.bn(H)
        return self.act(H)


class GraphEncoderSingle(nn.Module):
    """Stacked GCN encoder."""
    def __init__(self, in_dim: int, encoder_dim: int, encoder_depth: int):
        super().__init__()
        layers = [GCNLayer(in_dim, encoder_dim)]
        for _ in range(encoder_depth - 1):
            layers.append(GCNLayer(encoder_dim, encoder_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, X: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        H = X
        for layer in self.layers:
            H = layer(H, A)
        return H


class Projector(nn.Module):
    """2-layer MLP projector (BN + PReLU)."""
    def __init__(self, in_dim, proj_dim=128, hidden=256):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.act = nn.PReLU()
        self.lin2 = nn.Linear(hidden, proj_dim, bias=True)

    def forward(self, G: torch.Tensor) -> torch.Tensor:
        Z = self.lin1(G)
        Z = self.bn1(Z)
        Z = self.act(Z)
        return self.lin2(Z)


def global_readout(H_nodes: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """Global readout over nodes (mean/sum)."""
    return H_nodes.sum(dim=0) if mode == "sum" else H_nodes.mean(dim=0)


# ---------- Augmentations on Subgraph ----------

@dataclass
class AugParams:
    """Subgraph augmentation hyperparameters."""
    feat_mask_ratio: float = 0.0
    feat_noise_std: float = 0.0
    drop_edge_ratio: float = 0.0
    add_edge_ratio: float = 0.0
    drop_node_ratio: float = 0.0


def augment_graph(X: torch.Tensor, A: torch.Tensor, aug: AugParams) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply feature/edge/node perturbations to a subgraph."""
    n, C = X.shape
    X_aug = X.clone()
    A_aug = A.clone()

    if aug.feat_mask_ratio > 0:
        num_mask = int(C * aug.feat_mask_ratio)
        if num_mask > 0:
            idx = torch.randperm(C, device=X.device)[:num_mask]
            X_aug[:, idx] = 0.0

    if aug.feat_noise_std > 0:
        X_aug = X_aug + torch.randn_like(X_aug) * aug.feat_noise_std

    if aug.drop_node_ratio > 0:
        k = int(n * aug.drop_node_ratio)
        if k > 0:
            sel = torch.randperm(n, device=X.device)[:k]
            X_aug[sel, :] = 0.0
            A_aug[sel, :] = 0.0
            A_aug[:, sel] = 0.0
            for s in sel.tolist():
                A_aug[s, s] = 1.0

    if aug.drop_edge_ratio > 0:
        upper = torch.triu((A_aug > 0).float(), diagonal=1)
        pos = upper.nonzero(as_tuple=False)
        k = int(pos.shape[0] * aug.drop_edge_ratio)
        if k > 0 and pos.numel() > 0:
            sel = pos[torch.randperm(pos.shape[0], device=A.device)[:k]]
            for u, v in sel:
                A_aug[u, v] = 0.0
                A_aug[v, u] = 0.0

    if aug.add_edge_ratio > 0:
        upper0 = torch.triu((A_aug == 0).float(), diagonal=1)
        zero_pos = upper0.nonzero(as_tuple=False)
        k = int(zero_pos.shape[0] * aug.add_edge_ratio)
        if k > 0 and zero_pos.numel() > 0:
            sel = zero_pos[torch.randperm(zero_pos.shape[0], device=A.device)[:k]]
            for u, v in sel:
                A_aug[u, v] = 1.0
                A_aug[v, u] = 1.0

    A_aug = torch.maximum(A_aug, A_aug.t())
    A_aug = A_aug + torch.eye(n, device=A.device)
    return X_aug, A_aug


# ---------- InfoNCE + FNF ----------

def info_nce_bidir_with_fnf(z_w: torch.Tensor, z_s: torch.Tensor,
                            temperature: float = 0.2,
                            lambda_thresh: float = 0.9) -> torch.Tensor:
    """Bidirectional InfoNCE with false-negative filtering."""
    z_w = F.normalize(z_w, dim=-1)
    z_s = F.normalize(z_s, dim=-1)
    sim = z_w @ z_s.t()
    logits = sim / temperature
    B = z_w.size(0)
    labels = torch.arange(B, device=z_w.device)

    with torch.no_grad():
        eye = torch.eye(B, device=sim.device)
        neg_sim = sim * (1 - eye)
        fn_mask = (neg_sim > lambda_thresh)

    very_small = torch.finfo(logits.dtype).min
    loss1 = F.cross_entropy(logits.masked_fill(fn_mask, very_small), labels)
    loss2 = F.cross_entropy(logits.t().masked_fill(fn_mask.t(), very_small), labels)
    return 0.5 * (loss1 + loss2)


# ---------- Main Model ----------

class WSGraphCL(nn.Module):
    """Weak/Strong Graph Contrastive Learning with superpixels and K-hop subgraphs."""
    def __init__(self,
                 mode: str = 'linear_probe',
                 in_channels: int = 128,
                 encoder_dim: int = 128,
                 encoder_depth: int = 6,
                 proj_dim: int = 128,
                 proj_hidden: int = 256,
                 readout: str = "mean",
                 temperature: float = 0.2,
                 lambda_thresh: float = 0.9,
                 num_classes: int = 1000,
                 pretrained_path: Optional[str] = None,
                 sp_n_segments: int = 8,
                 sp_compactness: float = 10.0,
                 knn_k: int = 10,
                 eta_spatial: float = 0.5,
                 spectral_metric: str = "euclidean",
                 spatial_metric: str = "euclidean",
                 weight_mode: str = "heat",
                 heat_delta: float = 1.0,
                 khop_K: int = 1):
        super().__init__()
        self.encoder = GraphEncoderSingle(in_dim=in_channels,
                                          encoder_dim=encoder_dim,
                                          encoder_depth=encoder_depth)
        self.projector = Projector(encoder_dim, proj_dim=proj_dim, hidden=proj_hidden)
        self.cls_head = nn.Linear(encoder_dim, num_classes)

        self.readout = readout
        self.temperature = temperature
        self.lambda_thresh = lambda_thresh
        self.mode = mode

        self.sp_n_segments = sp_n_segments
        self.sp_compactness = sp_compactness
        self.knn_k = knn_k
        self.eta_spatial = eta_spatial
        self.spectral_metric = spectral_metric
        self.spatial_metric = spatial_metric
        self.weight_mode = weight_mode
        self.heat_delta = heat_delta
        self.khop_K = khop_K

        self.aug_weak = AugParams(0.05, 0.00, 0.00, 0.00, 0.00)
        self.aug_strong = AugParams(0.20, 0.05, 0.10, 0.05, 0.05)

        if pretrained_path is not None:
            self.load_pretrain(pretrained_path)

    def freeze_backbone(self):
        """Freeze encoder and projector."""
        for module in [self.encoder, self.projector]:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

    def load_pretrain(self, path: str, map_location: str = "cpu", strict: bool = False) -> Dict[str, Any]:
        """Load encoder/projector weights from checkpoint."""
        ckpt = torch.load(path, map_location=map_location)
        state = ckpt.get("state_dict", ckpt)
        own = self.state_dict()
        load_dict = {}
        for k, v in state.items():
            if k.startswith("encoder.") or k.startswith("projector."):
                if k in own and own[k].shape == v.shape:
                    load_dict[k] = v
        missing, unexpected = self.load_state_dict(load_dict, strict=False)
        return {"loaded_keys": list(load_dict.keys()), "missing": missing, "unexpected": unexpected}

    def _build_graph_from_bchw(self, x_c_hw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Build superpixel graph from a single (C,H,W) patch."""
        seg, n_sp = slic_superpixels(x_c_hw,
                                     n_segments=self.sp_n_segments,
                                     compactness=self.sp_compactness,
                                     sigma=0.0)
        X_sp, P_sp = superpixel_features_and_positions(x_c_hw, seg, n_sp)
        A_sp = build_knn_adjacency(
            X_sp, P_sp,
            k=self.knn_k,
            eta_spatial=self.eta_spatial,
            spectral_metric=self.spectral_metric,
            spatial_metric=self.spatial_metric,
            weight_mode=self.weight_mode,
            heat_delta=self.heat_delta
        )
        return X_sp, A_sp, seg

    def _pretrain_single_from_cached(self, X_sp: torch.Tensor, A_sp: torch.Tensor) -> torch.Tensor:
        """Pretrain on cached graph with K-hop subgraphs and weak/strong views."""
        device = next(self.parameters()).device
        X_sp = X_sp.to(device).to(torch.float32)
        A_sp = A_sp.to(device).to(torch.float32)
        N = X_sp.size(0)
        if N == 0:
            return torch.tensor(0.0, device=device)

        A_bin = (A_sp > 0).float()
        anchors = list(range(N))
        if len(anchors) > self.max_anchors_per_patch:  # expects attribute defined by caller
            anchors = random.sample(anchors, self.max_anchors_per_patch)

        z_w_list, z_s_list = [], []
        for center in anchors:
            mask = k_hop_mask(A_bin, center, self.khop_K)
            idx = mask.nonzero(as_tuple=False).squeeze(-1)
            X_sub = X_sp[idx, :]
            A_sub = A_sp[idx][:, idx]

            Xw, Aw = augment_graph(X_sub, A_sub, self.aug_weak)
            Xs, As = augment_graph(X_sub, A_sub, self.aug_strong)

            Hw = self.encoder(Xw, Aw)
            Hs = self.encoder(Xs, As)
            Gw = global_readout(Hw, mode=self.readout).unsqueeze(0)
            Gs = global_readout(Hs, mode=self.readout).unsqueeze(0)
            z_w_list.append(Gw)
            z_s_list.append(Gs)

        Z_w = self.projector(torch.cat(z_w_list, dim=0))
        Z_s = self.projector(torch.cat(z_s_list, dim=0))
        return info_nce_bidir_with_fnf(Z_w.detach(), Z_s,
                                       temperature=self.temperature,
                                       lambda_thresh=self.lambda_thresh)

    def _pretrain_single(self, x_c_hw: torch.Tensor) -> torch.Tensor:
        """Pretrain on a single (C,H,W) patch via graph, K-hop views, and InfoNCE."""
        X_sp, A_sp, _ = self._build_graph_from_bchw(x_c_hw)
        N = X_sp.shape[0]
        if N == 0:
            return torch.tensor(0.0, device=x_c_hw.device)

        A_bin = (A_sp > 0).float()
        anchors = list(range(N))

        z_w_list, z_s_list = [], []
        for center in anchors:
            mask = k_hop_mask(A_bin, center, self.khop_K)
            idx = mask.nonzero(as_tuple=False).squeeze(-1)
            X_sub = X_sp[idx, :]
            A_sub = A_sp[idx][:, idx]

            Xw, Aw = augment_graph(X_sub, A_sub, self.aug_weak)
            Xs, As = augment_graph(X_sub, A_sub, self.aug_strong)

            Hw = self.encoder(Xw, Aw)
            Hs = self.encoder(Xs, As)
            Gw = global_readout(Hw, mode=self.readout).unsqueeze(0)
            Gs = global_readout(Hs, mode=self.readout).unsqueeze(0)

            z_w_list.append(Gw)
            z_s_list.append(Gs)

        Z_w = self.projector(torch.cat(z_w_list, dim=0))
        Z_s = self.projector(torch.cat(z_s_list, dim=0))
        return info_nce_bidir_with_fnf(Z_w.detach(), Z_s,
                                       temperature=self.temperature,
                                       lambda_thresh=self.lambda_thresh)

    def _supervised_single(self, x_c_hw: torch.Tensor) -> torch.Tensor:
        """Supervised readout from the center superpixel's K-hop subgraph."""
        C, H, W = x_c_hw.shape
        X_sp, A_sp, seg = self._build_graph_from_bchw(x_c_hw)
        if X_sp.shape[0] == 0:
            g = torch.zeros(self.encoder.layers[-1].lin.out_features, device=x_c_hw.device)
            return self.cls_head(g.unsqueeze(0)).squeeze(0)

        center_sid = int(seg[H // 2, W // 2])
        A_bin = (A_sp > 0).float()
        mask = k_hop_mask(A_bin, center_sid, self.khop_K)
        idx = mask.nonzero(as_tuple=False).squeeze(-1)

        X_sub = X_sp[idx, :]
        A_sub = A_sp[idx][:, idx]
        Hn = self.encoder(X_sub, A_sub)
        G = global_readout(Hn, mode=self.readout)
        return self.cls_head(G.unsqueeze(0)).squeeze(0)

    def forward(self, x_bchw: torch.Tensor):
        """Pretrain -> (loss, loss); Linear/finetune -> logits (B,num_classes)."""
        assert x_bchw.dim() == 4
        if self.mode == "pretrain":
            losses = [self._pretrain_single(x_bchw[b]) for b in range(x_bchw.size(0))]
            loss = torch.stack(losses).mean()
            return loss, loss
        elif self.mode in {"linear_probe", "finetune"}:
            if self.mode == 'linear_probe':
                self.freeze_backbone()
            logits = [self._supervised_single(x_bchw[b]).unsqueeze(0) for b in range(x_bchw.size(0))]
            return torch.cat(logits, dim=0)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")