import torch
import random
from itertools import product
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange


def get_3d_sincos_pos_embed(embed_dim, t_size, grid_size, cls_token=False, scale_t=None):
    """3D (T,H,W) sine-cosine positional embedding, flattened to (T*H*W, D)."""
    assert embed_dim % 4 == 0
    embed_dim_spatial = embed_dim // 2
    embed_dim_temporal = embed_dim // 2

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # w first
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid)

    grid_t = np.arange(t_size, dtype=np.float32)
    pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t, scale=scale_t)

    pos_embed_temporal = pos_embed_temporal[:, np.newaxis, :]
    pos_embed_temporal = np.repeat(pos_embed_temporal, grid_size**2, axis=1)
    pos_embed_spatial = pos_embed_spatial[np.newaxis, :, :]
    pos_embed_spatial = np.repeat(pos_embed_spatial, t_size, axis=0)

    pos_embed = np.concatenate([pos_embed_temporal, pos_embed_spatial], axis=-1)
    pos_embed = pos_embed.reshape([-1, embed_dim])

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return torch.tensor(pos_embed, dtype=torch.float, requires_grad=False).unsqueeze(0)


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """2D sine-cosine positional embedding from a mesh grid."""
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos, scale=None):
    """1D sine-cosine positional embedding."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    if scale is not None:
        pos = pos * scale
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class PatchEmbed(nn.Module):
    """3D conv patch embedding for HSI (bands × H × W → tokens)."""
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        bands=32,
        b_patch_size=8,
        in_chans=1,
        embed_dim=768,
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        assert img_size[1] % patch_size[1] == 0
        assert img_size[0] % patch_size[0] == 0
        assert bands % b_patch_size == 0
        num_patches = (
            (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (bands // b_patch_size)
        )
        self.input_size = (
            bands // b_patch_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        print(f"img_size {img_size} patch_size {patch_size} bands {bands} b_patch_size {b_patch_size}")
        self.img_size = img_size
        self.patch_size = patch_size
        self.bands = bands
        self.b_patch_size = b_patch_size
        self.num_patches = num_patches
        self.grid_size = img_size[0] // patch_size[0]
        self.b_grid_size = bands // b_patch_size

        kernel_size = [b_patch_size] + list(patch_size)
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=kernel_size, stride=kernel_size)

    def forward(self, x):
        """Forward: (B, C=1, T=bands, H, W) → (B, T_grid, H*W, D)."""
        B, C, T, H, W = x.shape
        assert (H == self.img_size[0] and W == self.img_size[1]), \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        assert T == self.bands
        x = self.proj(x).flatten(3)
        x = torch.einsum("ncts->ntsc", x)
        self.output_size = x.shape
        return x


class Attention(nn.Module):
    """Multi-head attention with optional bias and drop."""
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        input_size=(4, 14, 14),
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        assert attn_drop == 0.0
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.input_size = input_size
        assert input_size[1] == input_size[2]

    def adjust_drop_rate(self, drop_rate=0.):
        """Dynamically adjust output dropout."""
        self.proj_drop = nn.Dropout(drop_rate)

    def forward(self, x, attn_bias=None):
        """Forward: (B,N,C) → (B,N,C)."""
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            attn += attn_bias
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x.view(B, -1, C)


class SwiGLU(nn.Module):
    """SwiGLU MLP with optional dropout."""
    def __init__(self, dim, hidden_dim, multiple_of=4, dropout=0.):
        super().__init__()
        hidden_dim = int(multiple_of * ((2 * hidden_dim // 3 + multiple_of - 1) // multiple_of))
        self.w1 = nn.Linear(dim, hidden_dim, bias=True)
        self.w2 = nn.Linear(hidden_dim, dim, bias=True)
        self.w3 = nn.Linear(dim, hidden_dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward: (B,N,C) → (B,N,C)."""
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Stochastic depth: randomly drop residual paths per-sample."""
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Module wrapper for stochastic depth."""
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class Block(nn.Module):
    """Transformer block: LN → MHA → DropPath → LN → SwiGLU → DropPath."""
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        attn_func=Attention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SwiGLU(dim, mlp_hidden_dim, mlp_ratio, drop)

    def forward(self, x):
        """Forward: (B,N,C) → (B,N,C)."""
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class HSIMAE(nn.Module):
    """HSIMAE for pretrain"""
    def __init__(
        self,
        img_size=7,
        patch_size=1,
        in_chans=1,
        embed_dim=128,
        depth=6,
        num_heads=8,
        decoder_embed_dim=64,
        decoder_depth=4,
        decoder_num_heads=8,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        bands=128,
        b_patch_size=32,
        no_qkv_bias=False,
        trunc_init=False,
        s_depth=3,
        mask_ratio=0.50
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.dim = embed_dim
        self.dec_dim = decoder_embed_dim
        self.s_depth = s_depth
        self.b_pred_patch_size = b_patch_size
        self.trunc_init = trunc_init
        self.norm_pix_loss = norm_pix_loss

        self.patch_embed = PatchEmbed(img_size, patch_size, bands, b_patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        if s_depth > 0:
            self.blocks_1 = nn.ModuleList([
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=not no_qkv_bias, norm_layer=norm_layer)
                for _ in range(s_depth)
            ])
            self.blocks_2 = nn.ModuleList([
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=not no_qkv_bias, norm_layer=norm_layer)
                for _ in range(s_depth)
            ])

        if s_depth < 12:
            self.blocks = nn.ModuleList([
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=not no_qkv_bias, norm_layer=norm_layer)
                for _ in range(s_depth, depth)
            ])
        self.norm = norm_layer(embed_dim)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=not no_qkv_bias, norm_layer=norm_layer)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, self.b_pred_patch_size * patch_size ** 2 * in_chans, bias=True
        )

        self.initialize_weights()
        print("model initialized")

    def initialize_weights(self):
        """Initialize positional embeddings and linear/layernorm params."""
        pos_embed = get_3d_sincos_pos_embed(self.dim, self.input_size[0], self.input_size[1])
        self.pos_embed.data.copy_(pos_embed)
        decoder_pos_embed = get_3d_sincos_pos_embed(self.dec_dim, self.input_size[0], self.input_size[1])
        self.decoder_pos_embed.data.copy_(decoder_pos_embed)
        self.pos_embed.requires_grad = False
        self.decoder_pos_embed.requires_grad = False

        w = self.patch_embed.proj.weight.data
        if self.trunc_init:
            torch.nn.init.trunc_normal_(w)
            torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        else:
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            torch.nn.init.normal_(self.mask_token, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Xavier/Trunc init for Linear; unit-gain for LayerNorm."""
        if isinstance(m, nn.Linear):
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """Convert images to patch tensors used for reconstruction target."""
        N, _, T, H, W = imgs.shape
        p = self.patch_embed.patch_size[0]
        u = self.b_pred_patch_size
        assert H == W and H % p == 0 and T % u == 0
        h = w = H // p
        t = T // u
        x = imgs.reshape(shape=(N, 1, t, u, h, p, w, p))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * p**2 * 1))
        self.patch_info = (N, T, H, W, p, u, t, h, w)
        return x

    def unpatchify(self, x):
        """Inverse of patchify to reconstruct images."""
        N, T, H, W, p, u, t, h, w = self.patch_info
        x = x.reshape(shape=(N, t, h, w, u, p, p, 1))
        x = torch.einsum("nthwupqc->nctuhpwq", x)
        imgs = x.reshape(shape=(N, 1, T, H, W))
        return imgs

    def get_dim_patches(self, T, L, mask_ratio):
        """Choose (len_t, len_l) whose product is closest to keep-ratio * T * L."""
        len_all = torch.tensor(list(product(range(2, T), range(2, L))))
        len_keep = (1 - mask_ratio) * T * L
        lens = len_all[:, 0] * len_all[:, 1]
        lens_diff = abs(len_keep - lens)
        ind = torch.where(lens_diff == torch.min(lens_diff))[0]
        r = torch.LongTensor(random.sample(range(len(ind)), 1))[0]
        index = len_all[ind[r]]
        len_t, len_l = index
        return len_t, len_l

    def spatial_spectral_masking(self, x, T, L, mask_ratio):
        """Joint spectral-spatial masking; returns masked tokens and indices."""
        N, _, D = x.shape
        mask_1 = torch.ones([N, T * L], device=x.device)
        mask_2 = torch.ones([N, T * L], device=x.device)

        self.len_t, self.len_l = self.get_dim_patches(T, L, mask_ratio)
        len_keep_1 = self.len_t * L
        len_keep_2 = self.len_l * T
        len_keep = self.len_t * self.len_l

        noise_1 = torch.rand(N, T, device=x.device).repeat_interleave(L, 1)
        ids_shuffle = torch.argsort(noise_1, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask_1[:, :len_keep_1] = 0
        mask_1 = torch.gather(mask_1, dim=1, index=ids_restore)

        noise_2 = torch.rand(N, L, device=x.device).repeat(1, T)
        ids_shuffle = torch.argsort(noise_2, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        mask_2[:, :len_keep_2] = 0
        mask_2 = torch.gather(mask_2, dim=1, index=ids_restore)

        mask_all = mask_1 + mask_2 + torch.linspace(0, 0.5, T * L, device=x.device).unsqueeze(0).repeat(N, 1)
        ids_shuffle = torch.argsort(mask_all, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        mask = torch.ones([N, T * L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, x, mask_ratio):
        """Encode tokens with spatial–spectral masking and Transformer blocks."""
        x = self.patch_embed(x)
        N, T, L, C = x.shape
        x = x.reshape(N, T * L, C)
        x, mask, ids_restore, ids_keep = self.spatial_spectral_masking(x, T, L, mask_ratio)
        x = x.view(N, -1, C)

        pos_embed = self.pos_embed.expand(x.shape[0], -1, -1)
        pos_embed = torch.gather(pos_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]))
        x = x + pos_embed

        if self.s_depth > 0:
            x1 = rearrange(x, 'b (t l) c -> (b t) l c', t=self.len_t, l=self.len_l)
            x2 = rearrange(x, 'b (t l) c -> (b l) t c', t=self.len_t, l=self.len_l)
            for blk in self.blocks_1:
                x1 = blk(x1)
            for blk in self.blocks_2:
                x2 = blk(x2)
            x1 = rearrange(x1, '(b t) l c -> b (t l) c', b=N, t=self.len_t)
            x2 = rearrange(x2, '(b l) t c -> b (t l) c', b=N, l=self.len_l)
            x = x1 + x2

        if self.s_depth < 12:
            for blk in self.blocks:
                x = blk(x)

        x = self.norm(x)
        return x, mask, ids_restore, ids_keep

    def forward_decoder(self, x, ids_restore):
        """Decode to predict masked tokens."""
        N = x.shape[0]
        T = self.patch_embed.b_grid_size
        H = W = self.patch_embed.grid_size

        x = self.decoder_embed(x)
        C = x.shape[-1]

        mask_token = x.mean(1).unsqueeze(1)
        mask_tokens = mask_token.repeat(1, T * H * W - x.shape[1], 1)

        x_ = torch.cat([x, mask_tokens], dim=1)
        x_ = x_.view([N, T * H * W, C])
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2]))
        x = x_.view([N, T * H * W, C])

        x = x + self.decoder_pos_embed
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        return self.decoder_pred(x)

    def forward_loss(self, imgs, pred, mask):
        """Masked MSE reconstruction loss (on removed patches only)."""
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
            self.mean = mean
            self.var = (var + 1.0e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        mask = mask.view(loss.shape)
        return (loss * mask).sum() / mask.sum()

    def recons(self, mask, pred):
        """Unpatchify mask/pred to image space for visualization."""
        mask = mask.unsqueeze(2).repeat(1, 1, pred.shape[2])
        mask = self.unpatchify(mask)
        if self.norm_pix_loss:
            pred = pred * self.var + self.mean
        pred = self.unpatchify(pred)
        return mask, pred

    def forward(self, imgs):
        """Pretrain forward: returns (loss, reconstruction)."""
        imgs = imgs.unsqueeze(1)
        latent, mask, ids_restore, ids_keep = self.forward_encoder(imgs, self.mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        mask, pred = self.recons(mask, pred)
        return loss, pred


class HSIMAE_lp(nn.Module):
    """HSIMAE for finetune / linear-probe"""
    def __init__(
            self,
            mode='linear_probe',   # 'finetune' / 'linear_probe'
            img_size=7,
            patch_size=1,
            in_chans=1,
            embed_dim=128,
            depth=6,
            num_heads=4,
            mlp_ratio=4.0,
            norm_layer=nn.LayerNorm,
            bands=128,
            b_patch_size=32,
            no_qkv_bias=False,
            trunc_init=False,
            s_depth=3,
            drop_path=0.,
            num_class=8,
            pretrained_path=None
    ):
        super().__init__()
        self.trunc_init = trunc_init
        self.b_pred_patch_size = b_patch_size

        self.patch_embed = PatchEmbed(img_size, patch_size, bands, b_patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size
        self.mode = mode
        self.s_depth = s_depth
        self.dim = embed_dim

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        if s_depth > 0:
            self.blocks_1 = nn.ModuleList([
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=not no_qkv_bias,
                      qk_scale=None, norm_layer=norm_layer, drop_path=dpr[i])
                for i in range(s_depth)
            ])
            self.blocks_2 = nn.ModuleList([
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=not no_qkv_bias,
                      qk_scale=None, norm_layer=norm_layer, drop_path=dpr[i])
                for i in range(s_depth)
            ])

        if s_depth < 12:
            self.blocks = nn.ModuleList([
                Block(embed_dim, num_heads, mlp_ratio, qkv_bias=not no_qkv_bias,
                      qk_scale=None, norm_layer=norm_layer, drop_path=dpr[i])
                for i in range(s_depth, depth)
            ])

        self.norm = norm_layer(embed_dim)
        self.cls_head = nn.Linear(embed_dim * self.patch_embed.b_grid_size, num_class)

        self.initialize_weights()
        if pretrained_path:
            self.load_pretrain(pretrained_path)

    def _freeze_encoder(self):
        """Freeze patch embed and transformer blocks (for linear probe)."""
        for module in [self.patch_embed,
                       *getattr(self, 'blocks_1', []),
                       *getattr(self, 'blocks_2', []),
                       *getattr(self, 'blocks', [])]:
            if module is None:
                continue
            module.eval()
            for p in module.parameters():
                p.requires_grad = False
        if hasattr(self, 'pos_embed'):
            self.pos_embed.requires_grad = False
        print(f"HSIMAE weights frozen for {self.mode}.")

    def load_pretrain(self, pretrained_path: str):
        """Load checkpoint with non-strict key matching."""
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

    def initialize_weights(self):
        """Initialize positional embeddings and patch conv weights."""
        pos_embed = get_3d_sincos_pos_embed(self.dim, self.input_size[0], self.input_size[1])
        self.pos_embed.data.copy_(pos_embed)
        self.pos_embed.requires_grad = False
        w = self.patch_embed.proj.weight.data
        if self.trunc_init:
            torch.nn.init.trunc_normal_(w)
        else:
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Init Linear/LayerNorm layers."""
        if isinstance(m, nn.Linear):
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        """Encode tokens with two-axis blocks and global blocks."""
        x = self.patch_embed(x)
        N, T, L, C = x.shape
        x = x.view(N, -1, C) + self.pos_embed

        if self.s_depth > 0:
            x1 = rearrange(x, 'b (t l) c -> (b t) l c', t=T, l=L)
            x2 = rearrange(x, 'b (t l) c -> (b l) t c', t=T, l=L)
            for blk in self.blocks_1:
                x1 = blk(x1)
            for blk in self.blocks_2:
                x2 = blk(x2)
            x1 = rearrange(x1, '(b t) l c -> b (t l) c', b=N, t=T)
            x2 = rearrange(x2, '(b l) t c -> b (t l) c', b=N, l=L)
            x = x1 + x2

        if self.s_depth < 12:
            for blk in self.blocks:
                x = blk(x)

        return self.norm(x)

    def head(self, x, type='AGG'):
        """Pooling head: GAP or grouped aggregation, then linear classifier."""
        N, T, L, C = self.patch_embed.output_size
        if type == 'GAP':
            x = x.reshape(N, -1, C)
        elif type == 'AGG':
            x = x.reshape(N, T, -1, C)
            x = x.permute(0, 2, 1, 3).flatten(2)
        x = x.mean(1)
        pred = self.cls_head(x)
        return pred, x

    def forward(self, imgs):
        """Forward: logits for linear-probe/finetune."""
        if self.mode == "linear_probe":
            self._freeze_encoder()
        imgs = imgs.unsqueeze(1)
        latent = self.forward_encoder(imgs)
        pred, latent = self.head(latent)
        return pred