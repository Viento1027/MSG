import os
import glob
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset, random_split


def norm_global(x):
    """Min–max normalize over the whole cube."""
    x = x.astype(np.float32)
    min_val = np.min(x)
    max_val = np.max(x)
    return (x - min_val) / (max_val - min_val + 1e-8)


def norm_channel(x):
    """Per-channel min–max normalize."""
    x = x.astype(np.float32)
    min_val = x.min(axis=(0, 1), keepdims=True)
    max_val = x.max(axis=(0, 1), keepdims=True)
    return (x - min_val) / (max_val - min_val + 1e-8)


def GWPCA(x, nc=128, group=4, whiten=False):
    """Group-wise PCA over spectral dimension."""
    h, w, c = x.shape
    x = x.reshape(-1, c)
    x_split = np.array_split(x, group, axis=1)

    pca_data_list = []
    for i, data in enumerate(x_split):
        group_var = np.var(data, axis=0)
        zero_var_mask = group_var < 1e-6
        if np.any(zero_var_mask):
            noise = np.random.normal(0, 1e-6, data.shape)
            data = data + noise * zero_var_mask
        if np.isnan(data).any() or np.isinf(data).any():
            print(f"Warning: invalid values in group {i}; cleaning.")
            data = np.nan_to_num(data, nan=0.0, posinf=1e6, neginf=-1e6)
        try:
            pca = PCA(n_components=min(data.shape[1], nc // group), whiten=whiten, random_state=0)
            pca_data = pca.fit_transform(data)
            pca_data_list.append(pca_data)
        except Exception as e:
            print(f"PCA failed: {str(e)}; fallback to raw data.")
            pca_data_list.append(data)

    merged = np.concatenate(pca_data_list, axis=-1)
    return merged.reshape(h, w, -1)


def load_hsi_mat(hsi_path, gt_path=None, hsi_key='hsi', gt_key='gt', normalize=True):
    """Load HSI/GT from .mat, with optional global normalization."""
    mat = loadmat(hsi_path)
    hsi = mat[hsi_key]
    if normalize:
        hsi = norm_global(hsi)
    gt = loadmat(gt_path)[gt_key] if gt_path else None
    return hsi, gt


def hsi_augment(data):
    """Simple flip/rotate augmentation on (H,W,C)."""
    do_augment = np.random.random()
    if do_augment > 0.5:
        prob = np.random.random()
        if 0 <= prob <= 0.2:
            data = np.fliplr(data)
        elif 0.2 < prob <= 0.4:
            data = np.flipud(data)
        elif 0.4 < prob <= 0.6:
            data = np.rot90(data, k=1)
        elif 0.6 < prob <= 0.8:
            data = np.rot90(data, k=2)
        else:
            data = np.rot90(data, k=3)
    return data


def stratified_subsample(indices, labels, ratio, seed):
    """Stratified subsample indices by class ratio."""
    assert 0 < ratio <= 1.0
    rng = np.random.RandomState(seed)
    sampled_idx = []
    classes = np.unique(labels)
    for c in classes:
        idx_c = indices[labels[indices] == c]
        n = len(idx_c)
        if n == 0:
            continue
        m = max(1, int(n * ratio))
        sampled = rng.choice(idx_c, size=m, replace=False)
        sampled_idx.extend(sampled.tolist())
    return np.array(sampled_idx)


def random_subsample(indices, ratio, seed):
    """Uniform subsample indices by ratio."""
    if ratio >= 1.0:
        return indices
    rng = np.random.default_rng(seed)
    m = max(1, int(len(indices) * ratio))
    return rng.choice(indices, size=m, replace=False)


class HSI_Dataset_npy(Dataset):
    """Patch dataset from a single .npy HSI cube (unsupervised)."""
    def __init__(self,
                 npy_path,
                 patch_size=7,
                 stride=7,
                 data_aug=True,
                 normalize=True,
                 applyPCA=True):
        self.npy_path = npy_path
        self.patch_size = patch_size
        self.stride = stride
        self.data_aug = data_aug
        self.normalize = normalize

        hsi_mm = np.load(npy_path, mmap_mode='r')
        H, W, C = hsi_mm.shape
        pad = patch_size // 2
        if np.isnan(hsi_mm).any() or np.isinf(hsi_mm).any():
            hsi_mm = np.nan_to_num(hsi_mm, nan=0.0, posinf=1e6, neginf=-1e6)

        self.hsi = GWPCA(hsi_mm, nc=32, group=4, whiten=False) if applyPCA else hsi_mm

        if normalize:
            # robust per-channel min/max by percentiles
            self.min_c = np.percentile(self.hsi, 1, axis=(0, 1))
            self.max_c = np.percentile(self.hsi, 99, axis=(0, 1))

        self.positions = [(i + pad, j + pad)
                          for i in range(0, H, stride)
                          for j in range(0, W, stride)]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        i, j = self.positions[idx]
        ps = self.patch_size
        pad = ps // 2
        H, W, C = self.hsi.shape

        rows = np.arange(i - pad, i + pad + 1)
        cols = np.arange(j - pad, j + pad + 1)
        # reflect-pad indexing
        rows = np.where(rows < 0, -rows - 1, rows)
        rows = np.where(rows >= H, 2 * H - rows - 1, rows)
        cols = np.where(cols < 0, -cols - 1, cols)
        cols = np.where(cols >= W, 2 * W - cols - 1, cols)

        patch = self.hsi[rows[:, None], cols[None, :], :]
        if self.data_aug:
            patch = hsi_augment(patch)
        if self.normalize:
            patch = (patch - self.min_c) / (self.max_c - self.min_c + 1e-8)

        return torch.from_numpy(patch.astype(np.float32)).permute(2, 0, 1)


class HSI_Dataset_Ft(Dataset):
    """Supervised patch dataset with labels from HSI+GT."""
    def __init__(self,
                 hsi,
                 gt,
                 sample_size=7,
                 stride=1,
                 data_aug=True,
                 normalize=True,
                 applyPCA=True,
                 offset=0):
        pad = sample_size // 2

        if applyPCA:
            hsi = GWPCA(hsi, nc=128, group=4, whiten=False)
        if normalize:
            hsi = norm_channel(hsi)

        self.data_aug = data_aug
        self.sample_size = sample_size

        self.hsi_padded = np.pad(hsi.astype(np.float32),
                                 ((pad, pad), (pad, pad), (0, 0)),
                                 mode='reflect')

        if isinstance(stride, int):
            sh = sw = max(1, stride)
        else:
            sh = max(1, int(stride[0]))
            sw = max(1, int(stride[1]))

        if isinstance(offset, int):
            oh = ow = max(0, int(offset))
        else:
            oh = max(0, int(offset[0]))
            ow = max(0, int(offset[1]))

        oh = oh % sh
        ow = ow % sw

        sub = gt[oh::sh, ow::sw]
        sub_coords = np.argwhere(sub != 0)
        if sub_coords.size > 0:
            coords = sub_coords * np.array([sh, sw]) + np.array([oh, ow])
        else:
            print("Fail at stride=", stride)
            coords = np.argwhere(gt != 0)

        self.positions = [(int(i) + pad, int(j) + pad) for i, j in coords]
        self.labels = [(gt[int(i), int(j)] - 1) for i, j in coords]

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, idx):
        i, j = self.positions[idx]
        ps = self.sample_size
        patch = self.hsi_padded[i - ps // 2:i - ps // 2 + ps,
                                j - ps // 2:j - ps // 2 + ps, :]
        label = self.labels[idx]
        if self.data_aug:
            patch = hsi_augment(patch)
        data = torch.from_numpy(patch.transpose(2, 0, 1).astype('float32'))
        label = torch.tensor(label, dtype=torch.int64)
        return data, label


def build_multi_pretrain_loader_npy(
        base_folder,
        npy_pattern='*.npy',
        sample_size=9,
        stride=1,
        batch_size=64,
        shuffle=True,
        num_workers=0,
        print_total=False,
        normalize=True,
        applyPCA=False,
        pretrain_num=None,
        seed=0
):
    """Concatenate multiple .npy cubes into one pretrain DataLoader."""
    npy_paths = glob.glob(os.path.join(base_folder, '**', npy_pattern), recursive=True)
    data_aug = True
    if not npy_paths:
        raise ValueError(f"No .npy files found in {base_folder} with pattern {npy_pattern}")
    else:
        print(f"Found {len(npy_paths)} .npy files in {base_folder}")
    datasets = [HSI_Dataset_npy(p, sample_size, stride, data_aug, normalize, applyPCA)
                for p in npy_paths]

    multi_ds = ConcatDataset(datasets)
    if pretrain_num is not None:
        total_n = len(multi_ds)
        if pretrain_num < total_n:
            rng = np.random.RandomState(0)
            indices = rng.choice(total_n, size=pretrain_num, replace=False)
            multi_ds = Subset(multi_ds, indices)
        else:
            print(f"pretrain_num ({pretrain_num}) >= total samples ({total_n}), using all samples")

    g = torch.Generator()
    g.manual_seed(seed)
    loader = DataLoader(multi_ds, batch_size=batch_size,
                        shuffle=shuffle, num_workers=num_workers, generator=g)

    if print_total:
        print(f"Total pretrain samples: {len(multi_ds)}")
    del datasets, multi_ds
    torch.cuda.empty_cache()
    return loader


def build_finetune_loader(hsi_path, gt_path,
                          hsi_key='hsi', gt_key='gt',
                          patch_size=9, batch_size=64,
                          num_workers=0, stride=1,
                          shuffle=True,
                          normalize=True, applyPCA=True,
                          train_ratio=0.7, val_ratio=0.15,
                          data_aug=True, seed=0,
                          split_mode="random_stratified", sample_ratio=1.0):
    """Build supervised loaders with random or stratified splits + optional sampling."""
    hsi, gt = load_hsi_mat(hsi_path, gt_path, hsi_key, gt_key)
    dataset = HSI_Dataset_Ft(hsi, gt, patch_size, stride, data_aug, normalize, applyPCA)
    g = torch.Generator().manual_seed(seed)

    if split_mode == "random":
        total_len = len(dataset)
        train_len = int(train_ratio * total_len)
        val_len = int(val_ratio * total_len)
        test_len = total_len - train_len - val_len

        train_ds, val_ds, test_ds = random_split(dataset,
                                                 lengths=[train_len, val_len, test_len],
                                                 generator=g)

        train_idx = random_subsample(list(range(len(train_ds))), sample_ratio, seed)
        val_idx = random_subsample(list(range(len(val_ds))), sample_ratio, seed)
        test_idx = random_subsample(list(range(len(test_ds))), sample_ratio, seed)

        train_ds = Subset(train_ds, train_idx)
        val_ds = Subset(val_ds, val_idx)
        test_ds = Subset(test_ds, test_idx)

        print(f"[Random Split + Sampling] Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    elif split_mode == "random_stratified":
        labels = np.array(dataset.labels)
        indices = np.arange(len(labels))

        train_idx, tmp_idx, _, tmp_labels = train_test_split(
            indices, labels, train_size=round(train_ratio, 6), stratify=labels, random_state=0
        )

        val_ratio_adj = val_ratio / (1 - train_ratio)
        val_idx, test_idx, _, _ = train_test_split(
            tmp_idx, tmp_labels, test_size=round(1 - val_ratio_adj, 6),
            stratify=tmp_labels, random_state=0
        )

        train_idx = stratified_subsample(train_idx, labels, sample_ratio, seed)
        val_idx = stratified_subsample(val_idx, labels, sample_ratio, seed)
        test_idx = stratified_subsample(test_idx, labels, sample_ratio, seed)

        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(dataset, val_idx)
        test_ds = Subset(dataset, test_idx)

        print(f"[Stratified Split + Sampling] Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    else:
        raise ValueError(f"Unknown split_mode={split_mode}, must be 'random' or 'random_stratified'")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, generator=g)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=shuffle, num_workers=num_workers, generator=g)
    test_loader = DataLoader(test_ds, batch_size=batch_size,
                             shuffle=shuffle, num_workers=num_workers, generator=g)
    return train_loader, val_loader, test_loader