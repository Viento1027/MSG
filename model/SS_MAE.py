import torch
import torch.nn as nn
from timm.models.vision_transformer import Block


class SS_MAE(nn.Module):
    """SS-MAE with pretrain/finetune/linear-probe modes."""
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
                 mask_ratio=0.75,
                 num_classes=1000,
                 pretrained_path=None):
        super().__init__()
        self.mode = mode
        self.patch_size = patch_size
        self.img_size = img_size
        self.mask_ratio = mask_ratio if mode == 'pretrain' else 0.0

        # Shared shapes
        self.num_spatial_patches = (img_size // patch_size) ** 2
        self.num_spectral_patches = channel_number
        self.channel_number = channel_number

        # ----- Spatial branch -----
        self.spatial_patch_embed = nn.Conv2d(
            channel_number, encoder_dim, kernel_size=patch_size, stride=patch_size
        )
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, self.num_spatial_patches, encoder_dim))
        self.spatial_encoder = nn.ModuleList([
            Block(encoder_dim, encoder_heads, mlp_ratio=mlp_ratio, qkv_bias=True)
            for _ in range(encoder_depth)
        ])
        self.spatial_norm = nn.LayerNorm(encoder_dim)

        # ----- Spectral branch -----
        self.spectral_patch_embed = nn.Sequential(
            nn.Flatten(start_dim=2),                  # [B, C, H*W]
            nn.Linear(img_size * img_size, encoder_dim)  # [B, C, D]
        )
        self.spectral_pos_embed = nn.Parameter(torch.zeros(1, channel_number, encoder_dim))
        self.spectral_encoder = nn.ModuleList([
            Block(encoder_dim, encoder_heads, mlp_ratio=mlp_ratio, qkv_bias=True)
            for _ in range(encoder_depth)
        ])
        self.spectral_norm = nn.LayerNorm(encoder_dim)

        # ----- Pretrain decoders -----
        if mode == 'pretrain':
            # spatial decoder
            self.spatial_decoder_embed = nn.Linear(encoder_dim, decoder_dim)
            self.spatial_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
            self.spatial_decoder_pos = nn.Parameter(torch.zeros(1, self.num_spatial_patches, decoder_dim))
            self.spatial_decoder = nn.ModuleList([
                Block(decoder_dim, decoder_heads, mlp_ratio=mlp_ratio, qkv_bias=True)
                for _ in range(decoder_depth)
            ])
            self.spatial_decoder_norm = nn.LayerNorm(decoder_dim)
            self.spatial_decoder_pred = nn.Linear(decoder_dim, patch_size ** 2 * channel_number)

            # spectral decoder
            self.spectral_decoder_embed = nn.Linear(encoder_dim, decoder_dim)
            self.spectral_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
            self.spectral_decoder_pos = nn.Parameter(torch.zeros(1, channel_number, decoder_dim))
            self.spectral_decoder = nn.ModuleList([
                Block(decoder_dim, decoder_heads, mlp_ratio=mlp_ratio, qkv_bias=True)
                for _ in range(decoder_depth)
            ])
            self.spectral_decoder_norm = nn.LayerNorm(decoder_dim)
            self.spectral_decoder_pred = nn.Linear(decoder_dim, img_size * img_size)

        # ----- Downstream head -----
        else:
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.linear_head = nn.Linear(encoder_dim * 2, num_classes)

        self._init_weights()
        if pretrained_path:
            self.load_pretrain(pretrained_path)

    def _init_weights(self):
        """Initialize parameters (pos embeds, mask tokens, linear/conv layers)."""
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.spectral_pos_embed, std=0.02)
        if self.mode == 'pretrain':
            nn.init.normal_(self.spatial_mask_token, std=0.02)
            nn.init.normal_(self.spectral_mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def freeze_backbone(self):
        """Freeze both branches (embeds + encoders + pos embeddings)."""
        for module in [self.spatial_patch_embed, self.spatial_encoder,
                       self.spectral_patch_embed, self.spectral_encoder]:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False
        for p in [self.spatial_pos_embed, self.spectral_pos_embed]:
            p.requires_grad = False

    def load_pretrain(self, pretrained_path: str):
        """Load weights (non-strict, strips 'module.' prefixes)."""
        ckpt = torch.load(pretrained_path, map_location='cpu')
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

    def _random_masking(self, x):
        """Per-sample random token masking; returns masked tokens, mask, ids_restore."""
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_shuffle)
        return x_masked, mask, ids_restore

    def forward(self, imgs):
        """Pretrain -> (loss_total, (spa_pred, spec_pred)); Finetune/Linear -> logits."""
        if self.mode == 'pretrain':
            # ----- spatial branch -----
            spatial = self.spatial_patch_embed(imgs)                 # [B, D, h, w]
            spatial = spatial.flatten(2).permute(0, 2, 1)            # [B, N, D]
            spatial = spatial + self.spatial_pos_embed
            spatial_masked, spatial_mask, spa_ids = self._random_masking(spatial)
            for blk in self.spatial_encoder:
                spatial_masked = blk(spatial_masked)
            spatial_latent = self.spatial_norm(spatial_masked)

            spa_dec = self.spatial_decoder_embed(spatial_latent)
            mask_tokens = self.spatial_mask_token.repeat(spa_dec.shape[0],
                                                         spa_ids.shape[1] - spa_dec.shape[1], 1)
            spa_dec = torch.cat([spa_dec, mask_tokens], dim=1)
            spa_dec = torch.gather(spa_dec, 1, spa_ids.unsqueeze(-1).expand(-1, -1, spa_dec.shape[-1]))
            spa_dec = spa_dec + self.spatial_decoder_pos
            for blk in self.spatial_decoder:
                spa_dec = blk(spa_dec)
            spa_pred = self.spatial_decoder_pred(self.spatial_decoder_norm(spa_dec))

            # ----- spectral branch -----
            spectral = self.spectral_patch_embed(imgs)               # [B, C, D]
            spectral = spectral + self.spectral_pos_embed
            spectral_masked, spectral_mask, spec_ids = self._random_masking(spectral)
            for blk in self.spectral_encoder:
                spectral_masked = blk(spectral_masked)
            spectral_latent = self.spectral_norm(spectral_masked)

            spec_dec = self.spectral_decoder_embed(spectral_latent)
            mask_tokens = self.spectral_mask_token.repeat(spec_dec.shape[0],
                                                          spec_ids.shape[1] - spec_dec.shape[1], 1)
            spec_dec = torch.cat([spec_dec, mask_tokens], dim=1)
            spec_dec = torch.gather(spec_dec, 1, spec_ids.unsqueeze(-1).expand(-1, -1, spec_dec.shape[-1]))
            spec_dec = spec_dec + self.spectral_decoder_pos
            for blk in self.spectral_decoder:
                spec_dec = blk(spec_dec)
            spec_pred = self.spectral_decoder_pred(self.spectral_decoder_norm(spec_dec))

            # losses
            target_spatial = imgs.permute(0, 2, 3, 1).reshape(imgs.shape[0], -1, imgs.shape[1])
            loss_spa = (spa_pred - target_spatial).pow(2).mean(-1)
            loss_spa = (loss_spa * spatial_mask).sum() / spatial_mask.sum()

            target_spectral = imgs.permute(0, 2, 3, 1).flatten(1, 2).permute(0, 2, 1)
            loss_spec = (spec_pred - target_spectral).pow(2).mean(-1)
            loss_spec = (loss_spec * spectral_mask).sum() / spectral_mask.sum()

            return (loss_spa + loss_spec), (spa_pred, spec_pred)

        elif self.mode == 'finetune':
            # spatial features
            spatial = self.spatial_patch_embed(imgs).flatten(2).permute(0, 2, 1)
            spatial = spatial + self.spatial_pos_embed
            for blk in self.spatial_encoder:
                spatial = blk(spatial)
            spatial = self.spatial_norm(spatial.mean(dim=1))

            # spectral features
            spectral = self.spectral_patch_embed(imgs)
            spectral = spectral + self.spectral_pos_embed
            for blk in self.spectral_encoder:
                spectral = blk(spectral)
            spectral = self.spectral_norm(spectral.mean(dim=1))

            combined = torch.cat([spatial, spectral], dim=1)
            return self.linear_head(combined)

        else:  # linear_probe
            self.freeze_backbone()
            spatial = self.spatial_patch_embed(imgs).flatten(2).permute(0, 2, 1)
            spatial = spatial + self.spatial_pos_embed
            for blk in self.spatial_encoder:
                spatial = blk(spatial)
            spatial = self.spatial_norm(spatial.mean(dim=1))

            spectral = self.spectral_patch_embed(imgs)
            spectral = spectral + self.spectral_pos_embed
            for blk in self.spectral_encoder:
                spectral = blk(spectral)
            spectral = self.spectral_norm(spectral.mean(dim=1))

            combined = torch.cat([spatial, spectral], dim=1)
            return self.linear_head(combined)