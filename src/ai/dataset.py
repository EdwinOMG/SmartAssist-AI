# src/ai/dataset.py
import os
import numpy as np
import random
import torch
from torch.utils.data import Dataset

"""
Dataset expects data/pose_dataset/<class_name>/*.npy
Each .npy is a numpy array of shape (T, K, C) where:
 - T = frames (variable length)
 - K = number of keypoints (e.g. 33 for MediaPipe Pose)
 - C = channels per keypoint (x, y, z, visibility) - but z/visibility optional

Returns:
 - seq: FloatTensor (seq_len, K * C)  (padded / truncated to fixed length)
 - label: LongTensor (class index)
 - meta: dict with original length, filepath (optional)
"""

class PoseSequenceDataset(Dataset):
    def __init__(self, root_dir, sequence_length=64, channels_last=True, transform=None, classes=None):
        super().__init__()
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.samples = []  # list of (filepath, label)
        self.class_to_idx = {}
        self.idx_to_class = {}
        subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        if classes:
            # allow passing specific class order
            subdirs = [d for d in subdirs if d in classes]
        for i, cls in enumerate(subdirs):
            self.class_to_idx[cls] = i
            self.idx_to_class[i] = cls
            class_dir = os.path.join(root_dir, cls)
            for f in sorted(os.listdir(class_dir)):
                if f.endswith('.npy') or f.endswith('.npz'):
                    self.samples.append((os.path.join(class_dir, f), i))

        if len(self.samples) == 0:
            raise RuntimeError(f"No .npy samples found in {root_dir}. Expected structure: {root_dir}/<class>/*.npy")

    def __len__(self):
        return len(self.samples)

    def _load_file(self, path):
        # Supports .npy or .npz
        if path.endswith('.npz'):
            data = np.load(path)['arr_0']
        else:
            data = np.load(path)
        # Expect shape (T, K, C) or (T, K)
        if data.ndim == 2:
            # assume (T, K) -> add channel dim
            data = data[:, :, np.newaxis]
        return data.astype(np.float32)

    def _pad_or_truncate(self, seq):
        T = seq.shape[0]
        if T < self.sequence_length:
            pad_len = self.sequence_length - T
            pad_shape = (pad_len, seq.shape[1], seq.shape[2])
            pad = np.zeros(pad_shape, dtype=seq.dtype)
            seq = np.concatenate([seq, pad], axis=0)
        elif T > self.sequence_length:
            seq = seq[:self.sequence_length]
        return seq

    def _flatten(self, seq):
        # flatten keypoints channels: (T, K, C) -> (T, K*C)
        T, K, C = seq.shape
        return seq.reshape(T, K * C)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        seq = self._load_file(path)  # (T, K, C)
        orig_len = seq.shape[0]

        # optional simple augmentation: random temporal jitter / small noise
        if self.transform:
            seq = self.transform(seq)

        seq = self._pad_or_truncate(seq)
        seq_flat = self._flatten(seq)  # (sequence_length, feat_dim)
        # normalize per-sample: translate by midpoint hip/pelvis and scale by torso length
        seq_norm = self._normalize_sequence(seq_flat, orig_seq=seq)

        return torch.from_numpy(seq_norm).float(), torch.tensor(label, dtype=torch.long), {"path": path, "orig_len": orig_len}

    def _normalize_sequence(self, flat_seq, orig_seq=None):
        # flat_seq: (T, K*C). We'll compute simple normalization using orig_seq (T,K,C)
        if orig_seq is None:
            return flat_seq
        # try to find mid-hip (MediaPipe ~ landmark 0 is nose; mid-hip index varies)
        # we will attempt common indices: use landmarks 23 & 24 (left/right hip) for MediaPipe
        try:
            kp = orig_seq  # (T, K, C)
            # attempt hip indices 23 & 24 (0-based)
            left_hip = kp[:, 23, :2]
            right_hip = kp[:, 24, :2]
            pelvis = (left_hip + right_hip) / 2.0  # (T,2)
            # torso length: distance between mid-shoulder and pelvis
            left_sh = kp[:, 11, :2]
            right_sh = kp[:, 12, :2]
            shoulder = (left_sh + right_sh) / 2.0
            torso = np.linalg.norm(shoulder - pelvis, axis=1).mean()
            torso = float(torso) if torso > 1e-6 else 1.0
            # subtract pelvis, scale by torso
            T = kp.shape[0]
            K = kp.shape[1]
            C = kp.shape[2]
            coords = kp[:, :, :2]  # (T,K,2)
            coords = coords - pelvis[:, None, :]  # translate
            coords = coords / torso  # scale
            # if more channels (z,vis) preserve them relative to normalization
            if C > 2:
                extras = kp[:, :, 2:]  # (T,K,C-2)
                new = np.concatenate([coords, extras], axis=2)
            else:
                new = coords
            # flatten to (T, K*C_new)
            T, K, Cnew = new.shape
            return new.reshape(T, K * Cnew)
        except Exception:
            # fallback: simple global normalization (subtract mean, divide by std)
            mu = flat_seq.mean(axis=0, keepdims=True)
            sigma = flat_seq.std(axis=0, keepdims=True) + 1e-6
            return (flat_seq - mu) / sigma