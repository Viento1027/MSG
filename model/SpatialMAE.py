import torch
import torch.nn as nn
from timm.models.vision_transformer import Block


class SpatialMAE(nn.Module):
    """Spatial Masked Autoencoder"""
    def __init__(self,
                 mode='pretrain',            # 'pretrain' | 'finetune' | 'linear_probe'
                 channel_number=128,
                 img_size=7,
                 patch_size=1,
                 encoder_dim=128,
                 encoder_depth=4,
                 encoder_heads=4,
                 decoder_dim=64,
                 decoder_depth=2,
                 decoder_heads=4,
                 mlp_ratio=4.0,
                 mask_ratio=0.7,
                 num_classes=1000,
                 pretrained_path=None):
        super().__init__()
        self.mode = mode
        self.patch_size = patch_size
        self.img_size = img_size
        self.mask_ratio = mask_ratio if mode == 'pretrain' else 0.0
        self.num_patches = (img_size // patch_size) ** 2

        # Encoder
        self.patch_embed = nn.Conv2d(channel_number, encoder_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, encoder_dim))
        self.encoder = nn.ModuleList([
            Block(encoder_dim, encoder_heads, mlp_ratio=mlp_ratio, qkv_bias=True)
            for _ in range(encoder_depth)
        ])
        self.norm = nn.LayerNorm(encoder_dim)

        # Decoder / Heads
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

        if pretrained_path:
            self.load_pretrain(pretrained_path)

    def freeze_backbone(self):
        """Freeze convolutional patch embed and transformer encoder (incl. pos_embed)."""
        for module in [self.patch_embed, self.encoder]:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False
        self.pos_embed.requires_grad = False

    def load_pretrain(self, pretrained_path: str):
        """Load checkpoint with non-strict key match (handles DataParallel prefixes)."""
        ckpt = torch.load(pretrained_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)

        stripped_state = {k[len('module.'):] if k.startswith('module.') else k: v
                          for k, v in state_dict.items()}

        load_result = self.load_state_dict(stripped_state, strict=False)
        missing = load_result.missing_keys
        total_keys = len(self.state_dict())
        loaded_keys = total_keys - len(missing)

        print(f"Loaded keys: {loaded_keys}/{total_keys}")
        if missing:
            print(f"Missing keys ({len(missing)}):")
            for k in missing:
                print(f"  - {k}")

    def random_masking(self, x):
        """Randomly mask tokens and return (masked_tokens, mask, ids_restore)."""
        B, N, D = x.shape
        len_keep = int(N * (1 - self.mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))
        mask = torch.ones(B, N, device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, 1, ids_shuffle)  # back to original order
        return x_masked, mask, ids_restore

    def forward_encoder(self, x):
        """Encode token sequence with Transformer blocks + LN."""
        for blk in self.encoder:
            x = blk(x)
        return self.norm(x)

    def forward_decoder(self, x, ids_restore):
        """Decode full sequence by inserting mask tokens and predicting pixels."""
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
        """MSE reconstruction loss on masked tokens."""
        target = imgs.permute(0, 2, 3, 1).reshape(imgs.shape[0], -1, imgs.shape[1])
        loss = (pred - target).pow(2).mean(-1)
        return (loss * mask).sum() / mask.sum()

    def forward(self, imgs):
        """Forward for pretrain (loss,pred) or finetune/linear-probe (logits)."""
        if self.mode == 'linear_probe':
            self.freeze_backbone()

        x = self.patch_embed(imgs).flatten(2).permute(0, 2, 1)
        x = x + self.pos_embed

        if self.mode == 'pretrain':
            x_masked, mask, ids_restore = self.random_masking(x)
            latent = self.forward_encoder(x_masked)
            pred = self.forward_decoder(latent, ids_restore)
            return self.forward_loss(imgs, pred, mask), pred

        feats = self.forward_encoder(x)
        pooled = self.global_pool(feats.transpose(1, 2)).flatten(1)
        return self.linear_head(pooled)