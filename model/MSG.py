import math
import torch
import torch.nn as nn


# =========================================
# Positional Encodings
# =========================================
def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w):
    """
    Create 2D sine-cosine positional embeddings.
    Return: (grid_size_h * grid_size_w, embed_dim)
    """

    def get_1d_embed(dim, pos):
        omega = torch.arange(dim // 2, dtype=torch.float32) / (dim / 2.0)
        omega = 1.0 / (10000 ** omega)          # [dim/2]
        pos = pos.flatten().unsqueeze(1)        # [M, 1]
        args = pos * omega.unsqueeze(0)         # [M, dim/2]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)  # [M, dim]
        return emb

    h_positions = torch.arange(int(grid_size_h), dtype=torch.float32)
    w_positions = torch.arange(int(grid_size_w), dtype=torch.float32)

    grid_h, grid_w = torch.meshgrid(h_positions, w_positions, indexing='ij')  # [H, W]
    emb_h = get_1d_embed(embed_dim // 2, grid_h)  # [H*W, D/2]
    emb_w = get_1d_embed(embed_dim // 2, grid_w)  # [H*W, D/2]
    return torch.cat([emb_h, emb_w], dim=1)       # [H*W, D]


def get_sincos_group_embedding(num_groups, dim):
    """
    Create 1D sine-cosine embeddings for group indices.
    Return: (num_groups, dim)
    """
    position = torch.arange(num_groups, dtype=torch.float32).unsqueeze(1)  # [G, 1]
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))  # [D/2]
    pe = torch.zeros(num_groups, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# =========================================
# Patch Embedding (Spectral Group) & Patchify
# =========================================
class PatchEmbedSpectralStem(nn.Module):
    """
    Spectral-stem + grouped patch projection.
    Input : (B, C, H, W)
    Output: (B, N, D) with N = num_groups * num_patches_per_group
    """

    def __init__(self, img_size, patch_size, channel_number, embed_dim, group_size):
        super().__init__()
        cal_ch = 2 * channel_number
        self.stem = nn.Sequential(
            nn.Conv2d(channel_number, cal_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(cal_ch),
            nn.GELU()
        )
        self.new_ch = cal_ch
        self.patch_size = patch_size
        self.group_size = group_size
        self.num_groups = self.new_ch // group_size // 2
        self.num_patches_per_group = (img_size // patch_size) ** 2
        self.num_patches = self.num_groups * self.num_patches_per_group

        self.proj = nn.Conv2d(
            self.new_ch,
            self.num_groups * embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            groups=self.num_groups
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.stem(x)
        x = self.proj(x)  # (B, num_groups*embed_dim, H/ps, W/ps)
        B, C2, H2, W2 = x.shape
        D = C2 // self.num_groups
        x = x.reshape(B, self.num_groups, D, H2, W2)        # (B, G, D, H2, W2)
        x = x.permute(0, 1, 3, 4, 2).reshape(B, self.num_patches, D)  # (B, N, D)
        return x


def patchify(imgs, patch_size, group_size):
    """
    Group-wise patchify.
    Input : (B, C, H, W)
    Output: (B, num_patches, patch_size^2 * group_size)
    """
    B, C, H, W = imgs.shape
    pad = (group_size - (C % group_size)) % group_size
    if pad:
        imgs = torch.cat(
            [imgs, torch.zeros(B, pad, H, W, device=imgs.device, dtype=imgs.dtype)],
            dim=1
        )
    new_C = imgs.shape[1]
    num_groups = new_C // group_size

    imgs = imgs.reshape(B, num_groups, group_size, H, W)
    h = H // patch_size
    w = W // patch_size
    imgs = imgs.reshape(B, num_groups, group_size, h, patch_size, w, patch_size)
    imgs = imgs.permute(0, 1, 3, 5, 4, 6, 2)               # (B, G, h, w, ps, ps, g)
    imgs = imgs.reshape(B, num_groups * h * w, patch_size * patch_size * group_size)
    return imgs


# =========================================
# MSG - Masked Spectral Group Framework
# =========================================
class MSG(nn.Module):
    def __init__(self,
                 mode='pretrain',            # 'pretrain' / 'finetune' / 'linear_probe'
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
                 mask_ratio_contrast=0.3,
                 contrastive_temp=0.1,
                 group_size=16,
                 num_classes=8,
                 pretrained_path=None):
        super().__init__()
        self.mode = mode
        self.patch_size = patch_size
        self.group_size = group_size
        self.mask_ratio = mask_ratio if mode == 'pretrain' else 0.0
        self.mask_ratio_contrast = mask_ratio_contrast
        self.contrastive_temp = contrastive_temp
        shared_depth = int(encoder_depth // 2)
        tail_depth = encoder_depth - shared_depth

        # Patch embed
        self.patch_embed = PatchEmbedSpectralStem(
            img_size, patch_size, channel_number, encoder_dim, group_size
        )
        self.num_groups = self.patch_embed.num_groups
        self.num_patches_per_group = self.patch_embed.num_patches_per_group
        self.num_patches = self.patch_embed.num_patches
        self.feat_dim = self.num_groups * encoder_dim  # G * D

        # Positional encodings
        pos = get_2d_sincos_pos_embed(encoder_dim, img_size / patch_size, img_size / patch_size)
        self.register_buffer('pos_embed', pos.unsqueeze(0))
        group_embed = get_sincos_group_embedding(self.num_groups, encoder_dim)
        self.register_buffer('group_embed', group_embed)

        # Shared encoder
        encoder_layer_shared = nn.TransformerEncoderLayer(
            d_model=encoder_dim, nhead=encoder_heads,
            dim_feedforward=int(encoder_dim * mlp_ratio)
        )
        self.encoder_shared = nn.TransformerEncoder(encoder_layer_shared, num_layers=shared_depth)
        self.shared_norm = nn.LayerNorm(encoder_dim)

        # Branch: MAE tail
        encoder_layer_mae = nn.TransformerEncoderLayer(
            d_model=encoder_dim, nhead=encoder_heads,
            dim_feedforward=int(encoder_dim * mlp_ratio)
        )
        self.encoder_mae = nn.TransformerEncoder(encoder_layer_mae, num_layers=tail_depth)
        self.encoder_norm_mae = nn.LayerNorm(encoder_dim)

        # Branch: Contrastive tail
        encoder_layer_contrast = nn.TransformerEncoderLayer(
            d_model=encoder_dim, nhead=encoder_heads,
            dim_feedforward=int(encoder_dim * mlp_ratio)
        )
        self.encoder_contrast = nn.TransformerEncoder(encoder_layer_contrast, num_layers=tail_depth)
        self.encoder_norm_contrast = nn.LayerNorm(encoder_dim)

        if mode == 'pretrain':
            # MAE decoder
            self.decoder_embed = nn.Linear(encoder_dim, decoder_dim, bias=True)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
            self.decoder_pos = nn.Parameter(torch.zeros(1, self.num_patches, decoder_dim))
            nn.init.trunc_normal_(self.decoder_pos, std=0.02)

            dec_layer = nn.TransformerEncoderLayer(
                d_model=decoder_dim, nhead=decoder_heads,
                dim_feedforward=int(decoder_dim * mlp_ratio)
            )
            self.decoder = nn.TransformerEncoder(dec_layer, num_layers=decoder_depth)
            self.decoder_norm = nn.LayerNorm(decoder_dim)
            self.decoder_pred = nn.Linear(
                decoder_dim, patch_size * patch_size * group_size, bias=True
            )

            # Contrastive projector
            self.projector = nn.Sequential(
                nn.Linear(self.feat_dim, self.feat_dim * 2),
                nn.BatchNorm1d(self.feat_dim * 2),
                nn.ReLU(),
                nn.Linear(self.feat_dim * 2, self.feat_dim)
            )
        else:
            # Linear probe head (fuses both branches)
            self.linear_head = nn.Linear(2 * self.feat_dim, num_classes)

        if pretrained_path:
            self.load_pretrain(pretrained_path)

    def load_pretrain(self, pretrained_path: str):
        """
        Load pretrained checkpoint with non-strict key matching.
        """
        ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        sd = ckpt.get('state_dict', ckpt)
        stripped = {k.replace('module.', ''): v for k, v in sd.items()}
        load_result = self.load_state_dict(stripped, strict=False)

        missing = load_result.missing_keys
        total_keys = len(self.state_dict())
        loaded_keys = total_keys - len(missing)
        print(f"Loaded keys: {loaded_keys}/{total_keys}")
        if missing:
            print(f"Missing keys ({len(missing)}):")
            for k in missing:
                print(f"  - {k}")

    def freeze_backbone(self):
        """
        Freeze patch embed and both encoder branches.
        """
        for module in [self.patch_embed, self.encoder_shared, self.encoder_mae, self.encoder_contrast]:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

    def random_masking(self, x, mask_ratio):
        """
        Randomly mask patches within each spectral group.
        Input : x (B, N, D) with N = G*T
        Return: masked tokens, binary mask, restore indices
        """
        B, N, D = x.shape
        G, T = self.num_groups, self.num_patches_per_group
        x = x.view(B, G, T, D)
        T_keep = int(T * (1 - mask_ratio))

        noise = torch.rand(B, G, T, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=2)
        ids_restore = torch.argsort(ids_shuffle, dim=2)

        ids_keep = ids_shuffle[:, :, :T_keep]
        x_masked = torch.gather(x, 2, ids_keep.unsqueeze(-1).expand(-1, -1, -1, D)).view(B, G * T_keep, D)

        mask = torch.ones(B, G, T, device=x.device)
        mask[:, :, :T_keep] = 0
        mask = torch.gather(mask, 2, ids_restore).view(B, G * T)

        return x_masked, mask, ids_restore.view(B, G * T)

    def forward_encoder(self, x, branch='mae'):
        """
        Shared encoder + branch-specific tail.
        """
        y = self.encoder_shared(x)
        y = self.shared_norm(y)
        if branch == 'mae':
            y = self.encoder_mae(y)
            return self.encoder_norm_mae(y)
        else:
            y = self.encoder_contrast(y)
            return self.encoder_norm_contrast(y)

    def forward_decoder(self, x, ids_restore):
        """
        Reconstruct tokens for MAE branch using a Transformer decoder.
        """
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.size(0), ids_restore.size(1) - x.size(1), 1)
        x_full = torch.cat([x, mask_tokens], dim=1)
        x_full = torch.gather(x_full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, x.size(-1)))
        x_full = x_full + self.decoder_pos
        x = x_full.transpose(0, 1)
        x = self.decoder(x).transpose(0, 1)
        return self.decoder_pred(self.decoder_norm(x))

    @staticmethod
    def nt_xent_loss(z1, z2, temperature=0.1):
        """
        Standard NT-Xent contrastive loss.
        """
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temperature
        mask = torch.eye(2 * B, device=sim.device).bool()
        sim = sim.masked_fill(mask, -9e15)
        pos = (torch.arange(2 * B, device=sim.device) + B) % (2 * B)
        return nn.functional.cross_entropy(sim, pos)

    def forward_loss(self, imgs, pred, mask):
        """
        MSE reconstruction loss on masked tokens.
        """
        target = patchify(imgs, self.patch_size, self.group_size)
        loss = (pred - target).pow(2).mean(-1)
        return (loss * mask).sum() / mask.sum()

    def forward(self, imgs):
        """
        Forward for pretraining (MAE + contrastive) or linear probe.
        """
        if self.mode == 'linear_probe':
            self.freeze_backbone()

        B = imgs.size(0)
        x = self.patch_embed(imgs)            # [B, G*T, D]
        _, N, D = x.shape
        G = self.num_groups
        T = self.num_patches_per_group

        # Add positional encodings (spatial + group)
        x = x.view(B, G, T, D)
        x = x + self.pos_embed.unsqueeze(1)
        x = x + self.group_embed.unsqueeze(0).unsqueeze(2)
        x = x.view(B, G * T, D)

        if self.mode == 'pretrain':
            # ----- MAE Reconstruction
            x_m, mask_m, ids_restore_m = self.random_masking(x, self.mask_ratio)
            latent_mae = self.forward_encoder(x_m, branch='mae')
            pred = self.forward_decoder(latent_mae, ids_restore_m)
            loss_rec = self.forward_loss(imgs, pred, mask_m)

            # ----- Contrastive (two masked views)
            x1, _, _ = self.random_masking(x, self.mask_ratio_contrast)
            x2, _, _ = self.random_masking(x, self.mask_ratio_contrast)

            z1 = self.forward_encoder(x1, branch='contrast')
            z2 = self.forward_encoder(x2, branch='contrast')

            z1_grp = z1.view(B, G, -1, D).mean(dim=2).view(B, G * D)
            z2_grp = z2.view(B, G, -1, D).mean(dim=2).view(B, G * D)

            c1 = nn.functional.normalize(z1_grp, dim=1)
            c2 = nn.functional.normalize(z2_grp, dim=1)
            p1 = nn.functional.normalize(self.projector(c1), dim=1)
            p2 = nn.functional.normalize(self.projector(c2), dim=1)

            loss_contrast = self.nt_xent_loss(p1, p2, self.contrastive_temp)
            loss_total = loss_rec + loss_contrast
            return loss_total, pred

        else:
            # Linear probe: average tokens per group for both branches, then fuse
            feats_mae = self.forward_encoder(x, branch='mae')
            feats_con = self.forward_encoder(x, branch='contrast')

            grp_mae = feats_mae.view(B, G, T, D).mean(dim=2).view(B, G * D)
            grp_con = feats_con.view(B, G, T, D).mean(dim=2).view(B, G * D)

            fused = torch.cat([grp_mae, grp_con], dim=1)  # [B, 2*G*D]
            return self.linear_head(fused)
