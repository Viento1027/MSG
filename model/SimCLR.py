import torch
import torch.nn as nn
import torch.nn.functional as F


def get_2d_sincos_pos_embed(embed_dim, grid_size_h, grid_size_w):
    """Create 2D sine-cosine positional embeddings -> (H*W, D)."""
    def get_1d_embed(dim, pos):
        omega = torch.arange(dim // 2, dtype=torch.float32) / (dim / 2.0)
        omega = 1.0 / (10000 ** omega)
        pos = pos.flatten().unsqueeze(1)
        args = pos * omega.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)

    h_positions = torch.arange(grid_size_h, dtype=torch.float32)
    w_positions = torch.arange(grid_size_w, dtype=torch.float32)
    grid_h, grid_w = torch.meshgrid(h_positions, w_positions, indexing='ij')
    emb_h = get_1d_embed(embed_dim // 2, grid_h)
    emb_w = get_1d_embed(embed_dim // 2, grid_w)
    return torch.cat([emb_h, emb_w], dim=1)


def random_flip(x, p=0.5):
    """Random horizontal/vertical flips for (B,C,H,W)."""
    if torch.rand(1) < p:
        x = torch.flip(x, dims=[3])
    if torch.rand(1) < p:
        x = torch.flip(x, dims=[2])
    return x


def random_color_jitter(x, brightness=0.2, contrast=0.2, p=0.8):
    """Global brightness/contrast jitter (SimCLR-style) for HSI."""
    if torch.rand(1) < p:
        B, _, _, _ = x.shape
        c = torch.empty(B, 1, 1, 1, device=x.device).uniform_(1 - contrast, 1 + contrast)
        b = torch.empty(B, 1, 1, 1, device=x.device).uniform_(-brightness, brightness)
        x = x * c + b
    return x


def random_gaussian_blur(x, kernel_size=3, sigma=(0.1, 2.0), p=0.5):
    """Per-band 2D Gaussian blur for (B,C,H,W)."""
    if torch.rand(1) >= p:
        return x
    sigma_val = float(torch.empty(1).uniform_(sigma[0], sigma[1]))
    k = kernel_size
    coords = torch.arange(k, device=x.device) - (k - 1) / 2.0
    grid_x = coords.unsqueeze(0).repeat(k, 1)
    grid_y = grid_x.t()
    kernel_2d = torch.exp(-((grid_x ** 2 + grid_y ** 2) / (2 * sigma_val ** 2)))
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel = kernel_2d.view(1, 1, k, k).repeat(x.shape[1], 1, 1, 1)
    padding = k // 2
    return F.conv2d(x, kernel, padding=padding, groups=x.shape[1])


def random_gaussian_noise(x, std=0.05, p=1.0):
    """Add Gaussian noise in spectral-spatial domain."""
    if torch.rand(1) < p:
        x = x + torch.randn_like(x) * std
    return x


class SimCLR_ViT(nn.Module):
    """Simple SimCLR-style ViT for HSI with projection/linear heads."""
    def __init__(self,
                 mode='pretrain',          # 'pretrain' | 'linear_probe' | 'finetune'
                 in_channels=128,
                 img_size=7,
                 patch_size=1,
                 encoder_dim=128,
                 encoder_depth=6,
                 encoder_heads=4,
                 mlp_ratio=4.0,
                 projector_out=128,
                 temperature=0.1,
                 num_classes=8,
                 pretrained_path=None):
        super().__init__()
        self.mode = mode
        self.temperature = temperature

        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, encoder_dim,
                                     kernel_size=patch_size, stride=patch_size)

        # Positional embedding
        pos = get_2d_sincos_pos_embed(encoder_dim, img_size // patch_size, img_size // patch_size)
        self.register_buffer('pos_embed', pos.unsqueeze(0))  # [1, N, D]

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=encoder_dim, nhead=encoder_heads,
            dim_feedforward=int(encoder_dim * mlp_ratio),
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=encoder_depth)
        self.encoder_norm = nn.LayerNorm(encoder_dim)

        # Projection head
        self.projector = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim * 2),
            nn.BatchNorm1d(encoder_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(encoder_dim * 2, projector_out)
        )

        # Linear head
        if mode in ('linear_probe', 'finetune'):
            self.linear_head = nn.Linear(encoder_dim, num_classes)
        else:
            # still create for convenience if user switches mode later
            self.linear_head = nn.Linear(encoder_dim, num_classes)

        if pretrained_path:
            self.load_pretrain(pretrained_path)

    def freeze_backbone(self):
        """Freeze patch embed and encoder (for linear probe)."""
        for module in [self.patch_embed, self.encoder]:
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

    def load_pretrain(self, pretrained_path: str):
        """Load checkpoint with non-strict key matching."""
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

    def _augment(self, x):
        """Compose SimCLR augmentations for HSI."""
        x = random_flip(x, p=0.5)
        x = random_color_jitter(x, brightness=0.2, contrast=0.2, p=0.8)
        x = random_gaussian_blur(x, kernel_size=3, sigma=(0.1, 2.0), p=0.5)
        return x

    def forward_backbone(self, x):
        """Encode (B,C,H,W) to a global feature (B,D)."""
        x = self.patch_embed(x)
        B, D, Nh, Nw = x.shape
        x = x.flatten(2).transpose(1, 2)          # [B, N, D]
        x = x + self.pos_embed[:, :Nh * Nw, :]    # add 2D pos
        x = self.encoder(x)
        x = self.encoder_norm(x)
        return x.mean(dim=1)                      # global avg pool -> [B, D]

    @staticmethod
    def nt_xent_loss(z1, z2, temperature=0.1):
        """Standard NT-Xent contrastive loss."""
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temperature
        mask = torch.eye(2 * B, device=sim.device).bool()
        sim = sim.masked_fill(mask, -9e15)
        pos = (torch.arange(2 * B, device=sim.device) + B) % (2 * B)
        return nn.functional.cross_entropy(sim, pos)

    def forward(self, x):
        """Pretrain -> (loss, z1, z2); Finetune/Linear -> logits."""
        if self.mode == 'pretrain':
            x1 = self._augment(x)
            x2 = self._augment(x)
            h1 = self.forward_backbone(x1)
            h2 = self.forward_backbone(x2)
            z1 = self.projector(h1)
            z2 = self.projector(h2)
            loss = self.nt_xent_loss(z1, z2, self.temperature)
            return loss, z1

        elif self.mode == 'finetune':
            h = self.forward_backbone(x)
            logits = self.linear_head(h)
            return logits

        else:  # linear_probe
            self.freeze_backbone()
            h = self.forward_backbone(x)
            logits = self.linear_head(h)
            return logits


if __name__ == '__main__':
    B, C, H, W = 16, 128, 7, 7
    test_input = torch.randn(B, C, H, W)

    model_pr = SimCLR_ViT(mode='pretrain')
    model_lp = SimCLR_ViT(mode='linear_probe')

    loss, z1, z2 = model_pr(test_input)
    logits = model_lp(test_input)

    print("Loss:", float(loss))
    print("Z1:", z1.size())
    print("Z2:", z2.size())
    print("Logits:", logits.size())