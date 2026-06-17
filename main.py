# Broad Obesity-1 — Perturbation Prediction v9
# Memory-optimised | scGPT/Geneformer embeddings supported
# Constants inside Config dataclass (satisfies submission requirements)


# !crunch setup-notebook broad-obesity-1 aCUOYLtifZLYtgoSitwyeRPn --no-model --no-data


import crunch

# Load the Crunch Toolings
#crunch_tools = crunch.load_notebook()


import numpy as np
import pandas as pd
import anndata as ad
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
from scipy.stats import skew as scipy_skew      # vectorized skew — replaces pd.Series loop
from scipy.sparse import issparse
from dataclasses import dataclass, field
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import warnings, gc, json as _json
import scanpy as sc
from scipy.stats import skew as scipy_skew
import scanpy as sc
from scipy.stats import skew as scipy_skew

#warnings.filterwarnings("ignore")
#np.random.seed(42)
#torch.manual_seed(42)

#print(f"PyTorch : {torch.__version__}")
#print(f"CUDA    : {torch.cuda.is_available()}")
#print(f"Device  : {'cuda' if torch.cuda.is_available() else 'cpu'}")


@dataclass
class Config:
    """All hyper-parameters live here.  Instantiate inside each function."""

    # ── Paths ──────────────────────────────────────────────────────────────
    data_dir   : Path = Path("data/")
    model_dir  : Path = Path("resources/")
    output_dir : Path = Path("outputs/")
    embed_dir  : Path = Path("resources/embeddings/")

    # ── Architecture ───────────────────────────────────────────────────────
    d_model      : int   = 256
    n_heads      : int   = 8
    n_layers     : int   = 4
    resid_hidden : list  = field(default_factory=lambda: [512, 256])
    dropout      : float = 0.15

    # ── Training ───────────────────────────────────────────────────────────
    lr           : float = 8e-5
    weight_decay : float = 1e-3
    epochs       : int   = 1200
    batch_size   : int   = 32
    patience     : int   = 150
    n_ensemble   : int   = 16

    # ── Loss weights ───────────────────────────────────────────────────────
    hvg_extra_w    : float = 4.0
    pearson_loss_w : float = 0.3
    prop_loss_w    : float = 0.2
    std_loss_w     : float = 0.15

    # ── Retrieve-and-Shift ─────────────────────────────────────────────────
    cells_per_pert : int   = 100
    k_retrieve     : int   = 10
    noise_blend    : float = 0.10
    embed_weight   : float = 0.4     # fraction of similarity from bio-embedding

    # ── Entropy calibration ────────────────────────────────────────────────
    entropy_min : float = 0.05
    entropy_max : float = 1.0

    # ── Program columns ────────────────────────────────────────────────────
    program_cols : list = field(default_factory=lambda: ["pre_adipo", "adipo", "lipo", "other"])


# ─────────────────────────────────────────────────────────────────────────────
# Utility & helper functions
# ─────────────────────────────────────────────────────────────────────────────

def to_dense(X):
    return X.toarray().astype(np.float32) if issparse(X) else np.array(X, dtype=np.float32)


def build_gene_features(gene_list, cross_pert_profiles, ctrl_stats_scaled):
    n_perts = len(next(iter(cross_pert_profiles.values())))
    feats   = {}
    for gene in gene_list:
        profile  = cross_pert_profiles.get(gene, np.zeros(n_perts,  dtype=np.float32))
        ctrl_vec = ctrl_stats_scaled.get(gene,   np.zeros(5,        dtype=np.float32))
        feats[gene] = np.concatenate([profile, ctrl_vec]).astype(np.float32)
    return feats


def build_safe_embed_dict(embed_matrix, gene_to_idx, target_genes):
    """Maps each gene in target_genes to its embedding with mean fallback."""
    mean_vec  = embed_matrix.mean(axis=0).astype(np.float32)
    d, n_miss = {}, 0
    for g in target_genes:
        if g in gene_to_idx:
            d[g] = embed_matrix[gene_to_idx[g]].astype(np.float32)
        else:
            d[g] = mean_vec.copy()
            n_miss += 1
    if n_miss:
        print(f"  Embed dict: {n_miss}/{len(target_genes)} genes used mean fallback")
    return d


def load_gene_embeddings(gene_names: list, embed_file: Path, order_file: Path):
    """
    Load pre-computed embeddings (from extract_foundation_embeddings.py) and
    align them to gene_names order.
    Returns: (matrix ndarray (n_genes, dim), embed_dim int)
    """
    if not embed_file.exists():
        raise FileNotFoundError(
            f"Gene embedding file not found: {embed_file}\n"
            f"Run: python extract_foundation_embeddings.py "
            f"--gene_list data/obesity_challenge_1.h5ad --out_dir {embed_file.parent}"
        )
    matrix = np.load(str(embed_file)).astype(np.float32)
    with open(order_file) as f:
        embed_genes = [l.strip() for l in f if l.strip()]
    embed_idx = {g: i for i, g in enumerate(embed_genes)}
    embed_dim = matrix.shape[1]
    mean_vec  = matrix.mean(axis=0).astype(np.float32)

    aligned, n_miss = [], 0
    for g in gene_names:
        if g in embed_idx:
            aligned.append(matrix[embed_idx[g]])
        else:
            aligned.append(mean_vec)
            n_miss += 1
    result = np.vstack(aligned).astype(np.float32)
    norms  = np.linalg.norm(result, axis=1, keepdims=True)
    result = result / np.where(norms > 1e-8, norms, 1.0)
    if n_miss:
        print(f"  Embeddings: {n_miss}/{len(gene_names)} genes used mean fallback")
    print(f"  Loaded gene embeddings: {result.shape}")
    return result, embed_dim


def compute_outlier_dist_np(query_embed: np.ndarray, pert_embed_matrix: np.ndarray) -> float:
    """
    Minimum cosine distance from query gene to all training pert embeddings.
    High distance (close to 2) → gene is biologically dissimilar to every training pert.
    """
    q_n  = query_embed / (np.linalg.norm(query_embed) + 1e-8)
    k_n  = pert_embed_matrix / (np.linalg.norm(pert_embed_matrix, axis=1, keepdims=True) + 1e-8)
    sims = k_n @ q_n
    return float(1.0 - sims.max())


def pearson_cosine_loss_hvg(pred_delta, target_delta, X_PM_hvg_dev, hvg_idx_t):
    p   = pred_delta[:, hvg_idx_t] - X_PM_hvg_dev.unsqueeze(0)
    t   = target_delta[:, hvg_idx_t] - X_PM_hvg_dev.unsqueeze(0)
    return -F.cosine_similarity(p, t, dim=-1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Model: DeltaInterpolationNetV9
# ─────────────────────────────────────────────────────────────────────────────

class DeltaInterpolationNetV9(nn.Module):
    """
    Key improvements vs v8:
    1. ENRICHED KEYS  — cross-attention keys fuse expression + bio-embedding.
       Model can now find biologically-similar training perts even when
       expression profiles differ (fixes FOXC1-class failures).
    2. STD HEAD       — predicts per-HVG-gene std used in affine Retrieve-and-Shift.
    3. OUTLIER SCORE  — minimum embedding distance fed into confidence head
       (extra shrinkage for TRIM5-class out-of-distribution genes).
    """

    def __init__(self, query_dim: int, n_train_perts: int, n_genes: int,
                 n_hvg: int, d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 4, resid_hidden=None, dropout: float = 0.15,
                 gene_embed_dim: int = 512,
                 entropy_min: float = 0.05, entropy_max: float = 1.0):
        super().__init__()
        if resid_hidden is None:
            resid_hidden = [512, 256]
        self.n_genes     = n_genes
        self.n_hvg       = n_hvg
        self.d_model     = d_model
        self.n_perts     = n_train_perts
        self.entropy_min = entropy_min
        self.entropy_max = entropy_max

        # ── Biological embedding projection (query path) ───────────────────
        self.embed_proj = nn.Sequential(
            nn.Linear(gene_embed_dim, d_model), nn.LayerNorm(d_model), nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model, d_model),        nn.LayerNorm(d_model),
        )
        self.embed_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.LayerNorm(d_model), nn.GELU(),
            nn.Linear(d_model, 1),           nn.Sigmoid()
        )

        # ── Expression feature projection (query path) ─────────────────────
        self.query_proj = nn.Sequential(
            nn.Linear(query_dim, d_model), nn.LayerNorm(d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),   nn.LayerNorm(d_model),
        )

        # ── Enriched key projection (expression + embedding) — v9 core fix ─
        self.key_expr_proj  = nn.Sequential(
            nn.Linear(query_dim, d_model // 2),      nn.LayerNorm(d_model // 2))
        self.key_embed_proj = nn.Sequential(
            nn.Linear(gene_embed_dim, d_model // 2), nn.LayerNorm(d_model // 2))
        self.key_fusion     = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model))

        # ── Attention ─────────────────────────────────────────────────────
        self.attn_temp  = nn.Parameter(torch.ones(1) * 1.5)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)

        # ── Transformer encoder ────────────────────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # ── Delta heads ────────────────────────────────────────────────────
        self.gate       = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())
        resid_layers, prev = [], d_model
        for h in resid_hidden:
            resid_layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        resid_layers.append(nn.Linear(prev, n_train_perts))
        self.resid_mlp  = nn.Sequential(*resid_layers)

        # ── Std head (NEW): predicts per-HVG std for distribution matching ─
        self.std_head = nn.Sequential(
            nn.Linear(d_model, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, n_hvg),   nn.Softplus())

        # ── Confidence head: entropy + outlier_dist → confidence scalar ───
        self.confidence_head = nn.Sequential(
            nn.Linear(d_model + 2, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())

        # ── Proportion head ────────────────────────────────────────────────
        self.prop_head = nn.Sequential(
            nn.Linear(d_model, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 4),       nn.Softmax(dim=-1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        # Bias embed_gate toward expression initially (gate≈1 → trust expression)
        for layer in self.embed_gate:
            if isinstance(layer, nn.Linear) and layer.out_features == 1:
                nn.init.constant_(layer.bias, 2.0)

    def forward(self, query_feat, pert_key_feats, pert_key_embeds,
                training_deltas, gene_embed=None,
                X_PM_delta=None, outlier_dist=None):
        B       = query_feat.shape[0]
        n_perts = pert_key_feats.shape[0]

        # Query: fuse expression + bio embedding
        q_expr = self.query_proj(query_feat)
        if gene_embed is not None:
            q_bio   = self.embed_proj(gene_embed)
            alpha   = self.embed_gate(torch.cat([q_expr, q_bio], dim=-1))
            q_fused = alpha * q_expr + (1 - alpha) * q_bio
        else:
            q_fused = q_expr

        # Keys: fuse expression + bio embedding (v9 architectural fix)
        kv_raw = self.key_fusion(torch.cat(
            [self.key_expr_proj(pert_key_feats),
             self.key_embed_proj(pert_key_embeds)], dim=-1
        ))
        kv = kv_raw.unsqueeze(0).expand(B, -1, -1)

        # Cross-attention
        q            = (q_fused * self.attn_temp).unsqueeze(1)
        attn_out, attn_w = self.cross_attn(q, kv, kv)
        primary_w    = attn_w.squeeze(1)       # (B, n_perts)

        # Attention entropy [0,1] — high entropy → model uncertain
        log_n   = float(np.log(n_perts))
        entropy = -(primary_w * torch.log(primary_w + 1e-10)).sum(-1) / log_n

        ctx = self.encoder(attn_out).squeeze(1)

        # Primary delta (attention-weighted interpolation)
        td                 = training_deltas.unsqueeze(0).expand(B, -1, -1)
        pred_delta_primary = torch.bmm(primary_w.unsqueeze(1), td).squeeze(1)

        # Residual delta (MLP-weighted)
        resid_w          = F.softmax(self.resid_mlp(ctx), dim=-1)
        pred_delta_resid = torch.bmm(resid_w.unsqueeze(1), td).squeeze(1)

        gate       = self.gate(ctx)
        pred_delta = gate * pred_delta_primary + (1 - gate) * pred_delta_resid

        # Std prediction
        pred_hvg_std = self.std_head(ctx)

        # Confidence calibration
        od = outlier_dist if outlier_dist is not None else torch.zeros(B, device=ctx.device)
        confidence = self.confidence_head(
            torch.cat([ctx, entropy.unsqueeze(-1), od.unsqueeze(-1)], dim=-1))
        confidence = self.entropy_min + (self.entropy_max - self.entropy_min) * confidence

        if X_PM_delta is not None:
            xpm        = X_PM_delta.unsqueeze(0).expand(B, -1)
            pred_delta = confidence * pred_delta + (1.0 - confidence) * xpm

        return pred_delta, pred_hvg_std, self.prop_head(ctx), confidence


# ─────────────────────────────────────────────────────────────────────────────
# Retrieve-and-Shift v9
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_and_shift_v9(
    predicted_mean, predicted_hvg_std, hvg_idx,
    query_feat, query_embed,
    train_feat_matrix, train_embed_matrix,
    train_perts, pert_means, pert_stds,
    X_pert_all, pert_cell_idx, global_residual_std,
    k=10, n_cells=100, noise_blend=0.10, embed_weight=0.4, seed=None,
):
    """
    Hybrid KNN + affine shift using model-predicted HVG std.
    - KNN uses expression similarity + embedding similarity
    - Affine transform uses pred_hvg_std for HVG genes (from std_head)
    - Falls back to weighted source stds for non-HVG genes
    """
    rng = np.random.default_rng(seed)

    q_e      = query_feat / (np.linalg.norm(query_feat) + 1e-8)
    k_e      = train_feat_matrix / (np.linalg.norm(train_feat_matrix, axis=1, keepdims=True) + 1e-8)
    sim_expr = k_e @ q_e

    if query_embed is not None and train_embed_matrix is not None:
        q_b     = query_embed / (np.linalg.norm(query_embed) + 1e-8)
        k_b     = train_embed_matrix / (np.linalg.norm(train_embed_matrix, axis=1, keepdims=True) + 1e-8)
        sims    = (1 - embed_weight) * sim_expr + embed_weight * (k_b @ q_b)
    else:
        sims = sim_expr

    top_k       = np.argsort(sims)[::-1][:k]
    top_sims    = sims[top_k]
    weights     = np.exp(top_sims - top_sims.max()); weights /= weights.sum()
    cells_per_k = np.round(weights * n_cells).astype(int)
    cells_per_k[0] += n_cells - cells_per_k.sum()

    # Build predicted std: model prediction for HVG, weighted source for others
    pred_std_full = sum(
        w * pert_stds[train_perts[ki]] for w, ki in zip(weights, top_k)
    ).astype(np.float32)
    if predicted_hvg_std is not None:
        pred_std_full[hvg_idx] = predicted_hvg_std.astype(np.float32)
    pred_std_full = np.clip(pred_std_full, 1e-8, None)

    shifted = []
    for i, k_idx in enumerate(top_k):
        if cells_per_k[i] == 0:
            continue
        pname          = train_perts[k_idx]
        src_mean       = pert_means[pname]
        src_std        = np.clip(pert_stds[pname], 1e-4, None)
        rows           = pert_cell_idx[pname]
        chosen         = rng.choice(rows, size=cells_per_k[i],
                                    replace=(len(rows) < cells_per_k[i]))
        cells          = X_pert_all[chosen].astype(np.float32)
        cells_centered = cells - src_mean[None, :]
        cells          = predicted_mean[None, :] + cells_centered * (pred_std_full / src_std)[None, :]
        noise          = (rng.standard_normal(cells.shape).astype(np.float32)
                          * global_residual_std * noise_blend)
        shifted.append(np.clip(cells + noise, 0, None))
        del cells, cells_centered, noise

    result = np.vstack(shifted).astype(np.float32)
    n = n_cells
    if len(result) > n:   result = result[:n]
    elif len(result) < n: result = np.vstack([result, result[:n - len(result)]])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Single model training step
# ─────────────────────────────────────────────────────────────────────────────

def train_single_model_v9(
    X_tr, y_delta_tr, y_prop_tr, prop_mask_tr, gene_embeds_tr,
    pert_key_feats_t, pert_key_embeds_t, training_deltas_t,
    hvg_idx_t, X_PM_t, X_PM_delta_t, y_hvg_std_tr,
    cfg,                              # Config instance
    X_val=None, y_delta_val=None, gene_embeds_val=None, y_hvg_std_val=None,
    seed=0, dropout=0.15, verbose=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    n_genes = training_deltas_t.shape[1]
    n_perts = pert_key_feats_t.shape[0]
    n_embed = gene_embeds_tr.shape[1]
    n_hvg   = len(hvg_idx_t)

    net = DeltaInterpolationNetV9(
        query_dim=X_tr.shape[1], n_train_perts=n_perts, n_genes=n_genes,
        n_hvg=n_hvg, dropout=dropout, gene_embed_dim=n_embed,
        entropy_min=cfg.entropy_min, entropy_max=cfg.entropy_max,
    ).to(device)

    optimizer = optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=300, T_mult=2, eta_min=1e-6)

    gene_weights               = torch.ones(n_genes, device=device)
    gene_weights[hvg_idx_t]   += cfg.hvg_extra_w
    X_PM_t_dev                 = X_PM_t.to(device)
    X_PM_delta_t_dev           = X_PM_delta_t.to(device)
    X_PM_hvg_dev               = X_PM_t_dev[hvg_idx_t]

    Xt  = torch.tensor(X_tr,           dtype=torch.float32, device=device)
    yd  = torch.tensor(y_delta_tr,     dtype=torch.float32, device=device)
    yp  = torch.tensor(y_prop_tr,      dtype=torch.float32, device=device)
    pm  = torch.tensor(prop_mask_tr,   dtype=torch.bool,    device=device)
    Xe  = torch.tensor(gene_embeds_tr, dtype=torch.float32, device=device)
    ys  = torch.tensor(y_hvg_std_tr,   dtype=torch.float32, device=device)

    loader = DataLoader(TensorDataset(Xt, yd, yp, pm, Xe, ys),
                        batch_size=cfg.batch_size, shuffle=True, drop_last=False)

    if X_val is not None:
        Xv  = torch.tensor(X_val,           dtype=torch.float32, device=device)
        yvd = torch.tensor(y_delta_val,     dtype=torch.float32, device=device)
        Xev = torch.tensor(gene_embeds_val, dtype=torch.float32, device=device)
        ysv = torch.tensor(y_hvg_std_val,   dtype=torch.float32, device=device)

    best_val, best_state, no_improve = float("inf"), None, 0
    train_losses, val_pearson_list   = [], []

    for epoch in range(cfg.epochs):
        net.train()
        ep_loss = 0.0

        for xb, ydb, ypb, pmb, xeb, ysb in loader:
            optimizer.zero_grad()
            pred_d, pred_s, pred_p, conf = net(
                xb, pert_key_feats_t, pert_key_embeds_t,
                training_deltas_t, gene_embed=xeb,
                X_PM_delta=X_PM_delta_t_dev, outlier_dist=None)

            diff  = pred_d - ydb
            loss  = (gene_weights * diff.pow(2)).mean()
            loss += cfg.pearson_loss_w * pearson_cosine_loss_hvg(
                        pred_d, ydb, X_PM_hvg_dev, hvg_idx_t)
            if pmb.any():
                loss += cfg.prop_loss_w * F.mse_loss(pred_p[pmb], ypb[pmb])
            loss += cfg.std_loss_w * F.mse_loss(pred_s, ysb)

            with torch.no_grad():
                hvg_cos = F.cosine_similarity(
                    pred_d[:, hvg_idx_t] - X_PM_hvg_dev,
                    ydb[:,  hvg_idx_t]   - X_PM_hvg_dev, dim=-1)
            loss += 0.05 * F.mse_loss(conf, ((hvg_cos + 1.0) / 2.0).detach().unsqueeze(-1))

            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item() * len(xb)

        ep_loss /= len(Xt)
        scheduler.step()
        train_losses.append(ep_loss)

        if X_val is not None:
            net.eval()
            with torch.no_grad():
                vd, vs, _, _ = net(Xv, pert_key_feats_t, pert_key_embeds_t,
                                   training_deltas_t, gene_embed=Xev,
                                   X_PM_delta=X_PM_delta_t_dev)
                val_mse  = (gene_weights * (vd - yvd).pow(2)).mean().item()
                val_pcos = pearson_cosine_loss_hvg(vd, yvd, X_PM_hvg_dev, hvg_idx_t).item()
                val_std  = F.mse_loss(vs, ysv).item()
                val_loss = val_mse + cfg.pearson_loss_w * val_pcos + cfg.std_loss_w * val_std

                vd_np  = vd.cpu().numpy()
                yvd_np = yvd.cpu().numpy()
                hvg_np = hvg_idx_t.cpu().numpy()
                xpm_np = X_PM_t.numpy()
                pearsons = []
                for b in range(len(vd_np)):
                    pd_ = vd_np[b][hvg_np] - xpm_np[hvg_np]
                    td_ = yvd_np[b][hvg_np] - xpm_np[hvg_np]
                    if np.std(pd_) > 1e-10 and np.std(td_) > 1e-10:
                        pearsons.append(pearsonr(pd_, td_)[0])
                vp = float(np.mean(pearsons)) if pearsons else 0.0
                val_pearson_list.append(vp)

            if val_loss < best_val:
                best_val, no_improve = val_loss, 0
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= cfg.patience:
                    if verbose: print(f"  Early stop @ epoch {epoch+1}")
                    break

        if verbose and (epoch + 1) % 100 == 0:
            msg = f"  Epoch {epoch+1:4d} | train={ep_loss:.5f}"
            if X_val is not None:
                msg += f" | val={val_loss:.5f} | val_pearson={vp:.4f}"
            print(msg)

    if best_state: net.load_state_dict(best_state)
    net.eval()
    best_vp = max(val_pearson_list) if val_pearson_list else None
    return net, train_losses, best_vp


# ─────────────────────────────────────────────────────────────────────────────
# train()
# ─────────────────────────────────────────────────────────────────────────────

def train(data_directory_path: str, model_directory_path: str) -> None:


    cfg        = Config()
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir   = Path(data_directory_path)
    model_dir  = Path(model_directory_path)
    model_dir.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Load & QC ────────────────────────────────────────────────
    print("[train] Loading AnnData...")
    adata              = ad.read_h5ad(data_dir / "obesity_challenge_1.h5ad")
    program_proportion = pd.read_csv(data_dir / "program_proportion.csv")
    if "percent.mt" in adata.obs.columns:
        adata = adata[adata.obs["percent.mt"] < 20].copy()
    if "nFeature_RNA" in adata.obs.columns:
        adata = adata[adata.obs["nFeature_RNA"] > 200].copy()
    print(f"  After QC: {adata.shape}")

    train_perts = sorted(adata.obs[adata.obs["gene"] != "NC"]["gene"].unique().tolist())
    ctrl_mask   = adata.obs["gene"] == "NC"
    gene_names  = np.array(adata.var_names.tolist())
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    n_genes     = len(gene_names)

    # ── Stage 2: Ctrl stats → DELETE X_ctrl immediately ───────────────────
    # FIX: use scipy_skew (vectorized) instead of a pd.Series loop over 21k genes
    print("[train] Computing ctrl stats (vectorized)...")
    X_ctrl    = to_dense(adata[ctrl_mask].X)
    ctrl_mean = X_ctrl.mean(axis=0).astype(np.float32)
    ctrl_std  = X_ctrl.std(axis=0).astype(np.float32)
    ctrl_pct  = (X_ctrl > 0).mean(axis=0).astype(np.float32)
    ctrl_skew = scipy_skew(X_ctrl, axis=0).astype(np.float32)   # vectorized!
    ctrl_cv   = np.where(ctrl_mean > 1e-8, ctrl_std / ctrl_mean, 0.0).astype(np.float32)
    del X_ctrl; gc.collect()
    print("  X_ctrl freed.")

    # ── Stage 3: HVG in-place — NO adata.copy() ───────────────────────────
    # FIX: inplace=True only adds columns to adata.var, never copies the X matrix
    print("[train] Computing HVG (in-place, no copy)...")
    sc.pp.highly_variable_genes(adata, n_top_genes=1000, flavor="seurat", inplace=True)
    hvg_names = adata.var_names[adata.var["highly_variable"]].tolist()
    hvg_idx   = np.array([gene_to_idx[g] for g in hvg_names if g in gene_to_idx], dtype=np.int64)
    hvg_idx_t = torch.tensor(hvg_idx, dtype=torch.long)
    n_hvg     = len(hvg_idx)
    print(f"  HVG: {n_hvg} genes")

    # ── Stage 4: Per-pert stats ONE AT A TIME — never full dense X_pert_all
    # FIX: X_pert_all (73k×21k dense = ~6 GB) is NOT needed during training.
    # retrieve_and_shift is only called at infer/evaluate time (which reload data).
    print("[train] Per-pert statistics (one pert at a time — low memory)...")
    pert_means     = {}
    pert_stds      = {}
    y_hvg_std_list = []
    X_PM_acc       = np.zeros(n_genes, dtype=np.float64)
    n_total_cells  = 0

    for g in tqdm(train_perts, desc="  per-pert"):
        cells          = to_dense(adata[adata.obs["gene"] == g].X)  # ~100 × 21k = 8 MB peak
        pert_means[g]  = cells.mean(axis=0).astype(np.float32)
        pert_stds[g]   = cells.std(axis=0).astype(np.float32) + 1e-8
        y_hvg_std_list.append(cells[:, hvg_idx].std(axis=0).astype(np.float32))
        X_PM_acc      += cells.sum(axis=0)
        n_total_cells += len(cells)
        del cells; gc.collect()

    X_PM         = (X_PM_acc / n_total_cells).astype(np.float32); del X_PM_acc
    X_PM_delta   = X_PM - ctrl_mean
    X_PM_t       = torch.tensor(X_PM,       dtype=torch.float32)
    X_PM_delta_t = torch.tensor(X_PM_delta, dtype=torch.float32)
    y_hvg_std_np = np.vstack(y_hvg_std_list).astype(np.float32)
    del y_hvg_std_list; gc.collect()
    print(f"  y_hvg_std shape: {y_hvg_std_np.shape}")

    # global_residual_std: std(cells - mean(cells)) = std(cells) = pert_stds[g]
    # No X_pert_all needed — mean of per-pert stds is equivalent.
    global_residual_std = np.mean(
        [s - 1e-8 for s in pert_stds.values()], axis=0
    ).astype(np.float32)

    # ── Stage 5: Delete adata — everything needed is now in RAM/dicts ─────
    del adata; gc.collect()
    print("  adata freed.")

    # ── Feature engineering ────────────────────────────────────────────────
    delta_matrix = np.vstack([pert_means[g] - ctrl_mean for g in train_perts]).astype(np.float32)

    cross_pert_mat = delta_matrix.T
    cp_mu  = cross_pert_mat.mean(axis=1, keepdims=True)
    cp_sig = cross_pert_mat.std(axis=1, keepdims=True) + 1e-8
    cps    = ((cross_pert_mat - cp_mu) / cp_sig).astype(np.float32)
    cross_pert_profiles = {g: cps[gene_to_idx[g]] for g in gene_names}
    del cross_pert_mat, cps; gc.collect()

    ctrl_stats_raw    = np.column_stack([ctrl_mean, ctrl_std, ctrl_pct, ctrl_skew, ctrl_cv])
    ctrl_stats_scaler = StandardScaler()
    ctrl_stats_scl    = ctrl_stats_scaler.fit_transform(ctrl_stats_raw).astype(np.float32)
    ctrl_stats_scaled = {g: ctrl_stats_scl[gene_to_idx[g]] for g in gene_names}
    del ctrl_stats_raw, ctrl_stats_scl; gc.collect()

    gene_features = build_gene_features(train_perts, cross_pert_profiles, ctrl_stats_scaled)
    N_INPUT       = len(next(iter(gene_features.values())))
    X_train_np    = np.array([gene_features[g] for g in train_perts], dtype=np.float32)

    # ── Gene embeddings ────────────────────────────────────────────────────
    print("[train] Loading gene embeddings...")
    embed_file  = cfg.embed_dir / "gene_embeddings_combined.npy"
    order_file  = cfg.embed_dir / "gene_embedding_order.txt"
    try:
        all_embeds, gene_embed_dim = load_gene_embeddings(gene_names.tolist(), embed_file, order_file)
        embed_dict    = build_safe_embed_dict(all_embeds, gene_to_idx, train_perts)
        pert_embed_np = np.array([embed_dict[g] for g in train_perts], dtype=np.float32)
        use_embed     = True
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")
        gene_embed_dim = 512
        pert_embed_np  = np.zeros((len(train_perts), gene_embed_dim), np.float32)
        embed_dict     = {g: np.zeros(gene_embed_dim, np.float32) for g in train_perts}
        all_embeds     = None
        use_embed      = False

    # ── Labels ─────────────────────────────────────────────────────────────
    y_delta_np = delta_matrix
    y_prop_np  = np.zeros((len(train_perts), 4), dtype=np.float32)
    prop_mask  = np.zeros(len(train_perts), dtype=bool)
    for i, g in enumerate(train_perts):
        row = program_proportion[program_proportion["gene"] == g]
        if not row.empty:
            y_prop_np[i] = row.iloc[0][cfg.program_cols].values.astype(np.float32)
            prop_mask[i] = True

    # ── GPU tensors (move once) ────────────────────────────────────────────
    pert_key_feats_t  = torch.tensor(X_train_np,   dtype=torch.float32).to(device)
    pert_key_embeds_t = torch.tensor(pert_embed_np, dtype=torch.float32).to(device)
    training_deltas_t = torch.tensor(delta_matrix,  dtype=torch.float32).to(device)

    # ── Ensemble training ─────────────────────────────────────────────────
    print(f"[train] Training {cfg.n_ensemble} ensemble models...")
    ensemble_models, ensemble_weights = [], []
    dropout_rates = [0.10, 0.12, 0.15, 0.18, 0.20] * 4
    n_full   = cfg.n_ensemble // 2
    loo_step = max(1, len(train_perts) // max(1, cfg.n_ensemble - n_full))

    for i in range(cfg.n_ensemble):
        dr = dropout_rates[i % len(dropout_rates)]
        print(f"\n--- Model {i+1}/{cfg.n_ensemble}  seed={i}  dropout={dr} ---")

        if i < n_full:
            m, tl, _ = train_single_model_v9(
                X_train_np, y_delta_np, y_prop_np, prop_mask, pert_embed_np,
                pert_key_feats_t, pert_key_embeds_t, training_deltas_t,
                hvg_idx_t, X_PM_t, X_PM_delta_t, y_hvg_std_np,
                cfg, seed=i, dropout=dr, verbose=True)
            ensemble_weights.append(1.0)
        else:
            loo_idx  = ((i - n_full) * loo_step) % len(train_perts)
            val_mask = np.array([j == loo_idx for j in range(len(train_perts))])
            tr_mask  = ~val_mask
            m, tl, best_vp = train_single_model_v9(
                X_train_np[tr_mask], y_delta_np[tr_mask],
                y_prop_np[tr_mask],  prop_mask[tr_mask],
                pert_embed_np[tr_mask],
                pert_key_feats_t, pert_key_embeds_t, training_deltas_t,
                hvg_idx_t, X_PM_t, X_PM_delta_t, y_hvg_std_np[tr_mask],
                cfg,
                X_val=X_train_np[val_mask],   y_delta_val=y_delta_np[val_mask],
                gene_embeds_val=pert_embed_np[val_mask],
                y_hvg_std_val=y_hvg_std_np[val_mask],
                seed=i, dropout=dr, verbose=True)
            w = np.exp(best_vp) if best_vp is not None else 1.0
            ensemble_weights.append(float(w))
            if best_vp is not None:
                print(f"  val_pearson={best_vp:.4f} → weight={w:.4f}")

        ensemble_models.append(m)
        print(f"  Final train loss: {tl[-1]:.6f}")
        torch.cuda.empty_cache()

    ew = np.array(ensemble_weights, np.float32); ew /= ew.sum()
    print(f"\nEnsemble weights: {ew.tolist()}")

    # ── KNN proportion backup ─────────────────────────────────────────────
    tp_list, tp_perts = [], []
    for g in train_perts:
        row = program_proportion[program_proportion["gene"] == g]
        if not row.empty:
            tp_list.append(row.iloc[0][cfg.program_cols].values.astype(np.float32))
            tp_perts.append(g)
    train_props_matrix   = np.vstack(tp_list)
    train_deltas_for_knn = np.vstack([pert_means[g] - ctrl_mean for g in tp_perts]).astype(np.float32)
    knn_prop = NearestNeighbors(n_neighbors=5, metric="cosine")
    knn_prop.fit(train_deltas_for_knn)

    # ── Save checkpoint ────────────────────────────────────────────────────
    print("[train] Saving checkpoint...")
    torch.save(dict(
        ensemble_states      = [m.state_dict() for m in ensemble_models],
        ensemble_weights     = ew,
        n_input              = N_INPUT,
        gene_embed_dim       = gene_embed_dim,
        use_embed            = use_embed,
        n_hvg                = n_hvg,
        ctrl_stats_scaler    = ctrl_stats_scaler,
        ctrl_mean            = ctrl_mean,
        ctrl_std             = ctrl_std,
        ctrl_pct             = ctrl_pct,
        ctrl_skew            = ctrl_skew,
        ctrl_cv              = ctrl_cv,
        cross_pert_profiles  = cross_pert_profiles,
        ctrl_stats_scaled    = ctrl_stats_scaled,
        delta_matrix         = delta_matrix,
        train_feat_matrix    = X_train_np,
        pert_embed_dict      = embed_dict,
        pert_embed_np        = pert_embed_np,
        train_embed_matrix   = all_embeds,
        train_perts          = train_perts,
        gene_names           = gene_names,
        pert_means           = pert_means,
        pert_stds            = pert_stds,
        global_residual_std  = global_residual_std,
        knn_prop             = knn_prop,
        train_props_matrix   = train_props_matrix,
        train_deltas_for_knn = train_deltas_for_knn,
        X_PM                 = X_PM,
        X_PM_delta           = X_PM_delta,
        hvg_names            = hvg_names,
        hvg_idx              = hvg_idx,
    ), model_dir / "ensemble_v9.pt")
    print(f"[train] Done → {model_dir}/ensemble_v9.pt")


#print("train() defined.")


# ─────────────────────────────────────────────────────────────────────────────
# infer()
# ─────────────────────────────────────────────────────────────────────────────

def infer(
    data_directory_path: str,
    prediction_directory_path: str,
    prediction_h5ad_file_path: str,
    program_proportion_csv_file_path: str,
    model_directory_path: str,
    predict_perturbations: list,
    genes_to_predict: list,
):
    cfg    = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir  = Path(data_directory_path)
    model_dir = Path(model_directory_path)
    Path(prediction_directory_path).mkdir(parents=True, exist_ok=True)

    print("[infer] Loading v9 checkpoint...")
    ckpt = torch.load(model_dir / "ensemble_v9.pt", map_location="cpu", weights_only=False)

    ctrl_mean          = ckpt["ctrl_mean"]
    ctrl_stats_scaler  = ckpt["ctrl_stats_scaler"]
    train_perts        = ckpt["train_perts"]
    gene_names         = ckpt["gene_names"]
    cross_pert_profiles= ckpt["cross_pert_profiles"]
    ctrl_stats_scaled  = ckpt["ctrl_stats_scaled"]
    delta_matrix       = ckpt["delta_matrix"]
    train_feat_matrix  = ckpt["train_feat_matrix"]
    pert_embed_np      = ckpt["pert_embed_np"]
    pert_embed_dict    = ckpt.get("pert_embed_dict", {})
    train_embed_matrix = ckpt.get("train_embed_matrix", None)
    pert_means         = ckpt["pert_means"]
    pert_stds          = ckpt.get("pert_stds", None)
    global_residual_std= ckpt["global_residual_std"]
    knn_prop           = ckpt["knn_prop"]
    train_props_matrix = ckpt["train_props_matrix"]
    n_input            = ckpt["n_input"]
    gene_embed_dim     = ckpt["gene_embed_dim"]
    use_embed          = ckpt.get("use_embed", False)
    n_hvg              = ckpt["n_hvg"]
    X_PM_delta         = ckpt.get("X_PM_delta", None)
    ew                 = ckpt.get("ensemble_weights", None)
    hvg_idx            = ckpt.get("hvg_idx", None)
    gene_to_idx        = {g: i for i, g in enumerate(gene_names)}

    print("[infer] Loading pert cells for Retrieve-and-Shift...")
    _ar  = ad.read_h5ad(data_dir / "obesity_challenge_1.h5ad")
    _pm  = _ar.obs["gene"] != "NC"
    _ps  = _ar[_pm]
    X_pert_all   = to_dense(_ps.X)
    _pl          = _ps.obs["gene"].values
    pert_cell_idx= {g: np.where(_pl == g)[0] for g in train_perts}
    del _ar, _ps; gc.collect()

    if pert_stds is None:
        pert_stds = {g: X_pert_all[pert_cell_idx[g]].std(axis=0).astype(np.float32) + 1e-8
                     for g in train_perts}

    pert_key_feats_t  = torch.tensor(train_feat_matrix, dtype=torch.float32).to(device)
    pert_key_embeds_t = torch.tensor(pert_embed_np,     dtype=torch.float32).to(device)
    training_deltas_t = torch.tensor(delta_matrix,      dtype=torch.float32).to(device)
    X_PM_delta_t      = (torch.tensor(X_PM_delta, dtype=torch.float32).to(device)
                         if X_PM_delta is not None else None)

    ensemble_models = []
    for state in ckpt["ensemble_states"]:
        m = DeltaInterpolationNetV9(
            query_dim=n_input, n_train_perts=len(train_perts),
            n_genes=delta_matrix.shape[1], n_hvg=n_hvg, gene_embed_dim=gene_embed_dim,
        ).to(device)
        m.load_state_dict(state); m.eval()
        ensemble_models.append(m)
    if ew is None:
        ew = np.ones(len(ensemble_models), np.float32) / len(ensemble_models)
    print(f"  Restored {len(ensemble_models)} models.")

    # Build expression features for unseen genes
    unseen = [g for g in predict_perturbations if g not in cross_pert_profiles]
    if unseen:
        print(f"[infer] Building features for {len(unseen)} unseen genes...")
        _ad2 = ad.read_h5ad(data_dir / "obesity_challenge_1.h5ad")
        _c2  = _ad2[_ad2.obs["gene"] == "NC"]
        for gene in unseen:
            if gene in _ad2.var_names:
                col   = to_dense(_c2[:, gene].X).ravel()
                m_    = col.mean(); s_ = col.std()
                raw   = np.array([[m_, s_, (col>0).mean(),
                                   float(pd.Series(col).skew()),
                                   s_/m_ if m_>1e-8 else 0]], np.float32)
                cross_pert_profiles[gene] = np.zeros(len(train_perts), np.float32)
                ctrl_stats_scaled[gene]   = ctrl_stats_scaler.transform(raw)[0]
            else:
                cross_pert_profiles[gene] = np.zeros(len(train_perts), np.float32)
                ctrl_stats_scaled[gene]   = np.zeros(5, np.float32)
        del _ad2, _c2; gc.collect()

    gene_features    = build_gene_features(predict_perturbations, cross_pert_profiles, ctrl_stats_scaled)
    valid_pred_genes = [g for g in genes_to_predict if g in gene_to_idx]
    pred_col_idx     = [gene_to_idx[g] for g in valid_pred_genes]

    mean_emb_inf = (np.mean(list(pert_embed_dict.values()), axis=0).astype(np.float32)
                    if pert_embed_dict else np.zeros(gene_embed_dim, np.float32))
    def _get_embed(g):
        if use_embed and g in pert_embed_dict:          return pert_embed_dict[g]
        if use_embed and train_embed_matrix is not None and g in gene_to_idx:
            return train_embed_matrix[gene_to_idx[g]]
        return mean_emb_inf.copy()

    def _predict(gene):
        feat = torch.tensor(gene_features[gene], dtype=torch.float32).unsqueeze(0).to(device)
        emb  = torch.tensor(_get_embed(gene),    dtype=torch.float32).unsqueeze(0).to(device)
        od   = torch.tensor(
            [[compute_outlier_dist_np(_get_embed(gene), pert_embed_np)]],
            dtype=torch.float32).squeeze(-1).to(device)
        dp, sp, pp = [], [], []
        for m, w in zip(ensemble_models, ew):
            with torch.no_grad():
                d, s, p, _ = m(feat, pert_key_feats_t, pert_key_embeds_t,
                                training_deltas_t, gene_embed=emb,
                                X_PM_delta=X_PM_delta_t, outlier_dist=od)
            dp.append(d.cpu().numpy()[0] * w); sp.append(s.cpu().numpy()[0] * w)
            pp.append(p.cpu().numpy()[0] * w)
        pred_delta = np.sum(dp, axis=0).astype(np.float32)
        nn_prop    = np.clip(np.sum(pp, axis=0), 0, None); nn_prop /= nn_prop.sum()
        return ctrl_mean + pred_delta, np.sum(sp, axis=0).astype(np.float32), nn_prop

    def _knn_prop(pred_mean):
        d, i = knn_prop.kneighbors((pred_mean - ctrl_mean).reshape(1, -1))
        w = 1.0/(d[0]+1e-8); w /= w.sum()
        p = np.clip((train_props_matrix[i[0]]*w[:,None]).sum(axis=0), 0, None)
        return p / p.sum()

    def _blend_props(nn_prop, knn_p):
        bp = np.clip(0.5*nn_prop + 0.5*knn_p, 0, None)
        cs = bp[0]+bp[1]+bp[3]
        if cs > 1e-8: bp[0]/=cs; bp[1]/=cs; bp[3]/=cs
        else:         bp[0]=bp[1]=0.; bp[3]=1.
        bp[2] = min(bp[2], bp[1])
        return bp

    print(f"[infer] Predicting {len(predict_perturbations)} perturbations...")
    all_cells, all_obs, prop_rows = [], [], []
    for gene in tqdm(predict_perturbations, desc="Inference"):
        pred_mean, pred_hvg_std, nn_prop = _predict(gene)
        cells_full = retrieve_and_shift_v9(
            pred_mean, pred_hvg_std, hvg_idx,
            gene_features[gene], _get_embed(gene) if use_embed else None,
            train_feat_matrix, pert_embed_np,
            train_perts, pert_means, pert_stds, X_pert_all, pert_cell_idx,
            global_residual_std,
            k=cfg.k_retrieve, n_cells=cfg.cells_per_pert,
            noise_blend=cfg.noise_blend, embed_weight=cfg.embed_weight,
            seed=abs(hash(gene)) % (2**31),
        )
        all_cells.append(cells_full[:, pred_col_idx])
        all_obs.extend([gene] * cfg.cells_per_pert)

        bp = _blend_props(nn_prop, _knn_prop(pred_mean))
        prop_rows.append({"gene": gene, **dict(zip(cfg.program_cols, bp.tolist()))})

    pred_matrix = np.vstack(all_cells).astype(np.float32)
    pred_adata  = ad.AnnData(
        X=pred_matrix,
        obs=pd.DataFrame({"gene": all_obs}),
        var=pd.DataFrame(index=valid_pred_genes))
    assert not np.isnan(pred_adata.X).any() and not np.isinf(pred_adata.X).any()
    pred_adata.write_h5ad(prediction_h5ad_file_path)

    prop_df = pd.DataFrame(prop_rows)[["gene"] + cfg.program_cols]
    assert np.allclose(prop_df[["pre_adipo","adipo","other"]].sum(axis=1), 1.0, atol=1e-3)
    assert (prop_df["lipo"] <= prop_df["adipo"] + 1e-8).all()
    prop_df.to_csv(program_proportion_csv_file_path, index=False)
    print(f"[infer] Done. {pred_adata.shape}")


#print("infer() defined.")


# ─────────────────────────────────────────────────────────────────────────────
# evaluate()  — matches official scoring exactly
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    data_directory_path: str,
    model_directory_path: str,
    output_directory_path: str,
):


    cfg    = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir   = Path(data_directory_path)
    model_dir  = Path(model_directory_path)
    output_dir = Path(output_directory_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    BANDWIDTHS = [581.5, 1163.0, 2326.0, 4652.0, 9304.0]

    def _pearson_delta(pred_hvg, actual_hvg, xpm_hvg):
        pd_ = pred_hvg - xpm_hvg; ad_ = actual_hvg - xpm_hvg
        if np.std(pd_) < 1e-10 or np.std(ad_) < 1e-10: return 0.0
        return float(pearsonr(pd_, ad_)[0])

    def _mmd2(X, Y):
        X = X.astype(np.float32); Y = Y.astype(np.float32)
        N, M = len(X), len(Y)
        def mk(A, B):
            diff = A[:,None,:] - B[None,:,:]
            sqd  = (diff**2).sum(-1)
            return sum(np.exp(-sqd/(2*bw)) for bw in BANDWIDTHS)
        return float(mk(X,X).sum()/(N*N) + mk(Y,Y).sum()/(M*M) - 2*mk(X,Y).sum()/(N*M))

    def _l1_program(pred_p, true_p):
        R = ["pre_adipo","adipo","other"]
        l1_r  = float(np.abs(np.array([pred_p[c] for c in R]) - np.array([true_p[c] for c in R])).sum())
        la_pr = pred_p.get("lipo_adipo", pred_p["lipo"] / max(pred_p["adipo"],1e-8))
        la_tr = true_p.get("lipo_adipo", true_p["lipo"] / max(true_p["adipo"],1e-8))
        return 0.75 * l1_r + 0.25 * abs(la_pr - la_tr)

    # Load checkpoint
    print("[eval] Loading v9 checkpoint...")
    ckpt = torch.load(model_dir/"ensemble_v9.pt", map_location="cpu", weights_only=False)

    ctrl_mean          = ckpt["ctrl_mean"]
    ctrl_stats_scaler  = ckpt["ctrl_stats_scaler"]
    train_perts        = ckpt["train_perts"]
    gene_names         = ckpt["gene_names"]
    cross_pert_profiles= ckpt["cross_pert_profiles"]
    ctrl_stats_scaled  = ckpt["ctrl_stats_scaled"]
    delta_matrix       = ckpt["delta_matrix"]
    train_feat_matrix  = ckpt["train_feat_matrix"]
    pert_embed_np      = ckpt["pert_embed_np"]
    pert_embed_dict    = ckpt.get("pert_embed_dict", {})
    train_embed_matrix = ckpt.get("train_embed_matrix", None)
    pert_means         = ckpt["pert_means"]
    pert_stds          = ckpt.get("pert_stds", {})
    global_residual_std= ckpt["global_residual_std"]
    knn_prop           = ckpt["knn_prop"]
    train_props_matrix = ckpt["train_props_matrix"]
    n_input            = ckpt["n_input"]
    gene_embed_dim     = ckpt["gene_embed_dim"]
    use_embed          = ckpt.get("use_embed", False)
    n_hvg              = ckpt["n_hvg"]
    X_PM               = ckpt["X_PM"]
    X_PM_delta         = ckpt["X_PM_delta"]
    ew                 = ckpt.get("ensemble_weights", None)
    hvg_idx            = ckpt["hvg_idx"]
    gene_to_idx        = {g: i for i, g in enumerate(gene_names)}

    # Load pert cells
    print("[eval] Loading pert cells...")
    _ar  = ad.read_h5ad(data_dir/"obesity_challenge_1.h5ad")
    _pm  = _ar.obs["gene"] != "NC"
    _ps  = _ar[_pm]
    X_pert_all   = to_dense(_ps.X)
    _pl          = _ps.obs["gene"].values
    pert_cell_idx= {g: np.where(_pl==g)[0] for g in train_perts}
    del _ar, _ps; gc.collect()

    if not pert_stds:
        pert_stds = {g: X_pert_all[pert_cell_idx[g]].std(axis=0).astype(np.float32)+1e-8
                     for g in train_perts}

    # Load ground truth
    gtruth_path = data_dir / "obesity_challenge_1_local_gtruth.h5ad"
    gprops_path = data_dir / "program_proportion_local_gtruth.csv"
    if not gtruth_path.exists():
        print(f"[eval] Ground truth not found: {gtruth_path}\n  Skipping.")
        return
    gtruth       = ad.read_h5ad(gtruth_path)
    gtruth_props = pd.read_csv(gprops_path)
    eval_genes   = sorted([g for g in gtruth.obs["gene"].unique() if g != "NC"])
    print(f"  Eval genes: {eval_genes}")

    X_pert_gt      = to_dense(gtruth[gtruth.obs["gene"] != "NC"].X)
    pert_labels_gt = gtruth[gtruth.obs["gene"] != "NC"].obs["gene"].values
    gtruth_genes   = np.array(gtruth.var_names.tolist())

    # HVG for evaluation
    print("[eval] Computing X_PM and HVGs...")
    _atmp = ad.read_h5ad(data_dir/"obesity_challenge_1.h5ad")
    X_PM_eval = to_dense(_atmp[_atmp.obs["gene"] != "NC"].X).mean(axis=0).astype(np.float32)
    sc.pp.highly_variable_genes(_atmp, n_top_genes=1000, flavor="seurat", inplace=True)
    hvg_eval   = _atmp.var_names[_atmp.var["highly_variable"]].tolist()
    del _atmp; gc.collect()

    common_hvg      = [g for g in hvg_eval if g in gene_to_idx and g in set(gtruth_genes)]
    tr_hvg_idx      = [gene_to_idx[g]                             for g in common_hvg]
    gt_hvg_idx      = [np.where(gtruth_genes==g)[0][0]            for g in common_hvg]
    X_PM_hvg        = X_PM_eval[tr_hvg_idx]
    print(f"  HVG genes in both: {len(common_hvg)}")

    # Restore models
    pert_key_feats_t  = torch.tensor(train_feat_matrix, dtype=torch.float32).to(device)
    pert_key_embeds_t = torch.tensor(pert_embed_np,     dtype=torch.float32).to(device)
    training_deltas_t = torch.tensor(delta_matrix,      dtype=torch.float32).to(device)
    X_PM_delta_t      = torch.tensor(X_PM_delta, dtype=torch.float32).to(device)

    ensemble_models = []
    for state in ckpt["ensemble_states"]:
        m = DeltaInterpolationNetV9(
            query_dim=n_input, n_train_perts=len(train_perts),
            n_genes=delta_matrix.shape[1], n_hvg=n_hvg, gene_embed_dim=gene_embed_dim,
        ).to(device)
        m.load_state_dict(state); m.eval()
        ensemble_models.append(m)
    if ew is None: ew = np.ones(len(ensemble_models), np.float32)/len(ensemble_models)
    print(f"  Restored {len(ensemble_models)} models.")

    # Features for eval genes
    unseen_eval = [g for g in eval_genes if g not in cross_pert_profiles]
    if unseen_eval:
        _ad2 = ad.read_h5ad(data_dir/"obesity_challenge_1.h5ad")
        _c2  = _ad2[_ad2.obs["gene"]=="NC"]
        for gene in unseen_eval:
            if gene in _ad2.var_names:
                col  = to_dense(_c2[:,gene].X).ravel()
                m_   = col.mean(); s_ = col.std()
                raw  = np.array([[m_,s_,(col>0).mean(),float(pd.Series(col).skew()),
                                  s_/m_ if m_>1e-8 else 0]],np.float32)
                cross_pert_profiles[gene] = np.zeros(len(train_perts),np.float32)
                ctrl_stats_scaled[gene]   = ctrl_stats_scaler.transform(raw)[0]
            else:
                cross_pert_profiles[gene] = np.zeros(len(train_perts),np.float32)
                ctrl_stats_scaled[gene]   = np.zeros(5,np.float32)
        del _ad2, _c2; gc.collect()

    gef = build_gene_features(eval_genes, cross_pert_profiles, ctrl_stats_scaled)
    mean_emb_ev = (np.mean(list(pert_embed_dict.values()),axis=0).astype(np.float32)
                   if pert_embed_dict else np.zeros(gene_embed_dim,np.float32))
    def _get_e(g):
        if use_embed and g in pert_embed_dict: return pert_embed_dict[g]
        if use_embed and train_embed_matrix is not None and g in gene_to_idx:
            return train_embed_matrix[gene_to_idx[g]]
        return mean_emb_ev.copy()

    pearson_scores, mmd_scores, l1_scores = [], [], []
    print(f"\n[eval] Evaluating {len(eval_genes)} unseen perturbations...")

    for gene in tqdm(eval_genes, desc="Eval"):
        feat = torch.tensor(gef[gene],   dtype=torch.float32).unsqueeze(0).to(device)
        emb  = torch.tensor(_get_e(gene),dtype=torch.float32).unsqueeze(0).to(device)
        od   = torch.tensor(
            [[compute_outlier_dist_np(_get_e(gene), pert_embed_np)]],
            dtype=torch.float32).squeeze(-1).to(device)
        dp, sp, pp = [], [], []
        for m, w in zip(ensemble_models, ew):
            with torch.no_grad():
                d,s,p,_ = m(feat, pert_key_feats_t, pert_key_embeds_t,
                             training_deltas_t, gene_embed=emb,
                             X_PM_delta=X_PM_delta_t, outlier_dist=od)
            dp.append(d.cpu().numpy()[0]*w); sp.append(s.cpu().numpy()[0]*w)
            pp.append(p.cpu().numpy()[0]*w)

        pred_delta   = np.sum(dp, axis=0).astype(np.float32)
        pred_hvg_std = np.sum(sp, axis=0).astype(np.float32)
        pred_mean    = (ctrl_mean + pred_delta).astype(np.float32)

        pred_cells = retrieve_and_shift_v9(
            pred_mean, pred_hvg_std, hvg_idx,
            gef[gene], _get_e(gene) if use_embed else None,
            train_feat_matrix, pert_embed_np,
            train_perts, pert_means, pert_stds, X_pert_all, pert_cell_idx,
            global_residual_std, k=10, n_cells=100, noise_blend=0.10, seed=42)

        actual_cells = X_pert_gt[pert_labels_gt == gene]
        pearson_scores.append(_pearson_delta(
            pred_mean[tr_hvg_idx], actual_cells[:, gt_hvg_idx].mean(axis=0), X_PM_hvg))
        mmd_scores.append(_mmd2(pred_cells[:, tr_hvg_idx], actual_cells[:, gt_hvg_idx]))

        true_row = gtruth_props[gtruth_props["gene"] == gene]
        if not true_row.empty:
            nn_p = np.clip(np.sum(pp,axis=0),0,None); nn_p/=nn_p.sum()
            kd,ki = knn_prop.kneighbors((pred_mean-ctrl_mean).reshape(1,-1))
            kw = 1.0/(kd[0]+1e-8); kw/=kw.sum()
            kp = np.clip((train_props_matrix[ki[0]]*kw[:,None]).sum(axis=0),0,None); kp/=kp.sum()
            bp = np.clip(0.5*nn_p+0.5*kp,0,None)
            cs = bp[0]+bp[1]+bp[3]
            if cs>1e-8: bp[0]/=cs; bp[1]/=cs; bp[3]/=cs
            else:       bp[0]=bp[1]=0.; bp[3]=1.
            bp[2]=min(bp[2],bp[1])
            pred_d = dict(zip(cfg.program_cols, bp.tolist()))
            true_d = true_row.iloc[0][cfg.program_cols].to_dict()
            l1_scores.append(_l1_program(pred_d, true_d))

    # Results
    print()
    print("=" * 65)
    print("V9 LOCAL EVALUATION  (matching official scoring code)")
    print("=" * 65)
    print(f"HVG genes used  : {len(common_hvg)}")
    print(f"Genes evaluated : {len(eval_genes)}")
    print()
    print(f"Pearson Delta : {np.mean(pearson_scores):.4f} +/- {np.std(pearson_scores):.4f}  (higher=better)")
    print(f"MMD²          : {np.mean(mmd_scores):.6f} +/- {np.std(mmd_scores):.6f}  (lower=better)")
    if l1_scores:
        print(f"L1 Program    : {np.mean(l1_scores):.4f} +/- {np.std(l1_scores):.4f}  (lower=better)")
    print(f"v8 baseline   : Pearson=0.2778 | MMD²=0.028493 | L1=0.1199")
    print("=" * 65)

    ev_df = pd.DataFrame({
        "gene": eval_genes, "pearson": pearson_scores, "mmd2": mmd_scores,
        "l1": l1_scores if len(l1_scores)==len(eval_genes) else [float("nan")]*len(eval_genes),
    }).sort_values("pearson", ascending=False)
    print(ev_df.to_string(index=False))

    ev_df.to_csv(output_dir/"eval_v9.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, sc_, title, col in zip(
        axes, [pearson_scores, mmd_scores, l1_scores or [0]],
        ["Pearson Delta", "MMD²", "L1 Program"], ["steelblue","seagreen","salmon"]):
        ax.hist(sc_, bins=min(10,len(sc_)), edgecolor="black", color=col)
        ax.axvline(np.mean(sc_), color="red", linestyle="--", label=f"Mean={np.mean(sc_):.4f}")
        ax.set_title(title); ax.legend(fontsize=8)
    plt.suptitle("V9 Evaluation", fontsize=12); plt.tight_layout()
    plt.savefig(output_dir/"evaluation_v9.png", dpi=120)
    print(f"Plot saved: {output_dir}/evaluation_v9.png")

    return dict(pearson=float(np.mean(pearson_scores)),
                mmd2=float(np.mean(mmd_scores)),
                l1=float(np.mean(l1_scores)) if l1_scores else float("nan"),
                per_gene=ev_df)


#print("evaluate() defined.")


#crunch_tools.test()


# # ─────────────────────────────────────────────────────────────────────────────
# # RUN — Train
# # ─────────────────────────────────────────────────────────────────────────────
# _cfg = Config()
# train(
#     data_directory_path  = str(_cfg.data_dir),
#     model_directory_path = str(_cfg.model_dir),
# )


# # ─────────────────────────────────────────────────────────────────────────────
# # RUN — Infer
# # ─────────────────────────────────────────────────────────────────────────────
# _cfg = Config()
# _perts = pd.read_csv(_cfg.data_dir/"predict_perturbations.txt", header=None)[0].tolist()
# _gtp   = _cfg.data_dir / "genes_to_predict.txt"
# _genes = (pd.read_csv(_gtp, header=None)[0].tolist() if _gtp.exists()
#           else ad.read_h5ad(_cfg.data_dir/"obesity_challenge_1.h5ad").var_names.tolist())

# infer(
#     data_directory_path              = str(_cfg.data_dir),
#     prediction_directory_path        = str(_cfg.output_dir),
#     prediction_h5ad_file_path        = str(_cfg.output_dir/"prediction.h5ad"),
#     program_proportion_csv_file_path = str(_cfg.output_dir/"predict_program_proportion.csv"),
#     model_directory_path             = str(_cfg.model_dir),
#     predict_perturbations            = _perts,
#     genes_to_predict                 = _genes,
# )


# # ─────────────────────────────────────────────────────────────────────────────
# # RUN — Evaluate  (requires local ground-truth h5ad)
# # ─────────────────────────────────────────────────────────────────────────────
# _cfg = Config()
# results = evaluate(
#     data_directory_path   = str(_cfg.data_dir),
#     model_directory_path  = str(_cfg.model_dir),
#     output_directory_path = str(_cfg.output_dir),
# )
