import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Block


# ---------- Contrastive: InfoNCE ----------
def info_nce_loss(q: torch.Tensor, k: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """In-batch InfoNCE with positives on the diagonal."""
    q = F.normalize(q, dim=-1)
    k = F.normalize(k, dim=-1)
    logits = q @ k.t() / temperature                  # [B, B]
    labels = torch.arange(q.size(0), device=q.device)
    return F.cross_entropy(logits, labels)


# ---------- EMA helper ----------
@torch.no_grad()
def ema_update(q_module: nn.Module, k_module: nn.Module, m: float):
    """EMA update k = m*k + (1-m)*q, in-place over matching params."""
    for p_q, p_k in zip(q_module.parameters(), k_module.parameters()):
        p_k.data.mul_(m).add_(p_q.data, alpha=1.0 - m)


class TMAC(nn.Module):
    """
    TMAC (MAE + momentum contrast).
    - Pretrain: MAE reconstruction (spatial masking) + contrastive branch.
    - Finetune: train backbone + linear head.
    - Linear-probe: freeze backbone, train linear head.
    """

    def __init__(self,
                 mode='pretrain',            # 'pretrain' | 'finetune' | 'linear_probe'
                 channel_number=128,
                 img_size=7,
                 patch_size=1,
                 encoder_dim=128,
                 encoder_depth=6,
                 encoder_heads=4,
                 decoder_dim=64,
                 decoder_depth=2,
                 decoder_heads=4,
                 mlp_ratio=4.0,
                 mask_ratio=0.7,
                 num_classes=1000,
                 # contrastive / EMA
                 proj_dim=128,
                 temperature=0.2,
                 lambda_contrast=0.5,
                 momentum=0.99,
                 # light augmentations (pretrain only)
                 aug_hflip=True,
                 aug_vflip=True,
                 aug_channel_scale_std=0.02,
                 aug_noise_std=0.01,
                 pretrained_path=None):
        super().__init__()
        self.mode = mode
        self.patch_size = patch_size
        self.img_size = img_size
        self.mask_ratio = mask_ratio if mode == 'pretrain' else 0.0
        self.num_patches = (img_size // patch_size) ** 2

        # ----- Encoder -----
        self.patch_embed = nn.Conv2d(channel_number, encoder_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, encoder_dim))
        self.encoder = nn.ModuleList([
            Block(encoder_dim, encoder_heads, mlp_ratio=mlp_ratio, qkv_bias=True)
            for _ in range(encoder_depth)
        ])
        self.norm = nn.LayerNorm(encoder_dim)

        # ----- Decoder / Heads -----
        if mode == 'pretrain':
            self.decoder_embed = nn.Linear(encoder_dim, decoder_dim)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
            self.decoder_pos = nn.Parameter(torch.zeros(1, self.num_patches, decoder_dim))
            self.decoder = nn.ModuleList([
                Block(decoder_dim, decoder_heads, mlp_ratio=mlp_ratio, qkv_bias=True)
                for _ in range(decoder_depth)
            ])
            self.decoder_norm = nn.LayerNorm(decoder_dim)
            self.decoder_pred = nn.Linear(decoder_dim, patch_size ** 2 * channel_number)
        else:
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.linear_head = nn.Linear(encoder_dim, num_classes)

        # ----- Momentum branch (no grad) -----
        self.patch_embed_k = copy.deepcopy(self.patch_embed)
        self.pos_embed_k = nn.Parameter(self.pos_embed.detach().clone(), requires_grad=False)
        self.encoder_k = copy.deepcopy(self.encoder)
        self.norm_k = copy.deepcopy(self.norm)
        for m in [self.patch_embed_k, self.encoder_k, self.norm_k]:
            for p in m.parameters():
                p.requires_grad = False

        # Optional feature decoder for momentum branch (kept for parity)
        self.feature_decoder_k = nn.ModuleList([
            Block(encoder_dim, encoder_heads, mlp_ratio=mlp_ratio, qkv_bias=True)
            for _ in range(2)
        ])
        self.feature_decoder_k_norm = nn.LayerNorm(encoder_dim)

        # ----- Projectors (q/k) -----
        self.proj_q = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.GELU(),
            nn.Linear(encoder_dim, proj_dim),
        )
        self.proj_k = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.GELU(),
            nn.Linear(encoder_dim, proj_dim),
        )
        for p in self.proj_k.parameters():
            p.requires_grad = False

        # hyper-params
        self.temperature = temperature
        self.lambda_contrast = lambda_contrast
        self.momentum = momentum

        # augment cfg
        self.aug_hflip = aug_hflip
        self.aug_vflip = aug_vflip
        self.aug_channel_scale_std = aug_channel_scale_std
        self.aug_noise_std = aug_noise_std

        if pretrained_path:
            self.load_pretrain(pretrained_path)

    # ----- I/O helpers -----
    def load_pretrain(self, pretrained_path: str):
        """Load checkpoint with relaxed key matching."""
        ckpt = torch.load(pretrained_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        stripped = {k[len('module.'):] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        self.load_state_dict(stripped, strict=False)

    def freeze_backbone(self):
        """Freeze encoder for linear probe."""
        for module in [self.patch_embed, self.encoder]:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False
        self.pos_embed.requires_grad = False
        self.patch_embed_k.eval()
        self.norm_k.eval()
        for blk in self.encoder_k:
            blk.eval()

    # ----- Augmentations -----
    def _augment(self, imgs: torch.Tensor) -> torch.Tensor:
        """Light per-sample flips, per-channel scale, and Gaussian noise."""
        x = imgs
        if self.aug_hflip:
            mask = torch.rand(x.size(0), 1, 1, 1, device=x.device) < 0.5
            x = torch.where(mask, torch.flip(x, dims=[3]), x)
        if self.aug_vflip:
            mask = torch.rand(x.size(0), 1, 1, 1, device=x.device) < 0.5
            x = torch.where(mask, torch.flip(x, dims=[2]), x)
        if self.aug_channel_scale_std and self.aug_channel_scale_std > 0:
            scale = torch.randn(x.size(0), x.size(1), 1, 1, device=x.device) * self.aug_channel_scale_std + 1.0
            x = x * scale
        if self.aug_noise_std and self.aug_noise_std > 0:
            x = x + torch.randn_like(x) * self.aug_noise_std
        return x

    # ----- MAE core -----
    def random_masking(self, x):
        """Sample visible tokens and return (x_vis, mask, ids_restore, ids_keep)."""
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_vis = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_shuffle)
        return x_vis, mask, ids_restore, ids_keep

    def forward_encoder(self, x):
        """Encode visible tokens with ViT blocks + LN."""
        for blk in self.encoder:
            x = blk(x)
        return self.norm(x)

    def forward_decoder(self, x, ids_restore):
        """Decode to pixels from visible tokens + mask tokens."""
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(x, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        x = x + self.decoder_pos
        for blk in self.decoder:
            x = blk(x)
        x = self.decoder_norm(x)
        return self.decoder_pred(x)

    def forward_loss(self, imgs, pred, mask):
        """MSE over masked tokens."""
        target = imgs.permute(0, 2, 3, 1).reshape(imgs.shape[0], -1, imgs.shape[1])
        loss = (pred - target).pow(2).mean(-1)
        return (loss * mask).sum() / mask.sum()

    # ----- Momentum encoder on visible tokens -----
    @torch.no_grad()
    def _encode_visible_k(self, imgs, ids_keep):
        """Encode the same visible token positions with the momentum branch."""
        x_full = self.patch_embed_k(imgs).flatten(2).permute(0, 2, 1)
        x_full = x_full + self.pos_embed_k
        D = x_full.size(-1)
        x_vis = torch.gather(x_full, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        for blk in self.encoder_k:
            x_vis = blk(x_vis)
        x_vis = self.norm_k(x_vis)
        return x_vis.mean(dim=1)

    @torch.no_grad()
    def _ema_step(self):
        """Update momentum branch parameters with EMA."""
        m = self.momentum
        ema_update(self.patch_embed, self.patch_embed_k, m)
        self.pos_embed_k.data.mul_(m).add_(self.pos_embed.data, alpha=1.0 - m)
        ema_update(self.norm, self.norm_k, m)
        for blk_q, blk_k in zip(self.encoder, self.encoder_k):
            ema_update(blk_q, blk_k, m)
        ema_update(self.proj_q, self.proj_k, m)

    # ----- Downstream helpers -----
    def ensure_head(self, num_classes: int):
        """Create/resize linear head when switching dataset/classes."""
        if not hasattr(self, 'linear_head') or self.linear_head.out_features != num_classes:
            self.linear_head = nn.Linear(self.norm.normalized_shape[0], num_classes)

    def forward_features(self, imgs):
        """Tokenize + encode + global pool -> (B, D)."""
        x = self.patch_embed(imgs).flatten(2).permute(0, 2, 1)
        x = x + self.pos_embed
        feats = self.forward_encoder(x)
        pooled = self.global_pool(feats.transpose(1, 2)).squeeze(-1)
        return pooled

    # ----- Main forward -----
    def forward(self, imgs):
        """Pretrain -> (loss, pred); Finetune/Linear -> logits."""
        if self.mode == 'linear_probe':
            self.freeze_backbone()

        if self.mode == 'pretrain':
            imgs_view = self._augment(imgs)
            x_tokens = self.patch_embed(imgs_view).flatten(2).permute(0, 2, 1)
            x_tokens = x_tokens + self.pos_embed
            x_vis, mask, ids_restore, ids_keep = self.random_masking(x_tokens)

            latent_p = self.forward_encoder(x_vis)
            pred = self.forward_decoder(latent_p, ids_restore)
            loss_rec = self.forward_loss(imgs_view, pred, mask)

            q_vec = latent_p.mean(dim=1)
            q_proj = self.proj_q(q_vec)
            with torch.no_grad():
                k_vec = self._encode_visible_k(imgs_view, ids_keep)
                k_proj = self.proj_k(k_vec)

            loss_con = info_nce_loss(q_proj, k_proj, temperature=self.temperature)
            loss = loss_rec + self.lambda_contrast * loss_con

            self._ema_step()
            return loss, pred

        # finetune / linear_probe
        pooled = self.forward_features(imgs)
        return self.linear_head(pooled)
