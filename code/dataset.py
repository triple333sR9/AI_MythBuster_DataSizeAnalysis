"""Dataset loading and sampling utilities."""

from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split


class FashionMNISTDataset:
    """Fashion-MNIST with stratified sampling and noise injection."""

    def __init__(self, data_dir: str = "./data"):
        self.transform = transforms.Normalize((0.2860,), (0.3530,))
        train = datasets.FashionMNIST(data_dir, train=True, download=True)
        test = datasets.FashionMNIST(data_dir, train=False, download=True)
        self.train_data = train.data.numpy()
        self.train_labels = train.targets.numpy()
        self.test_data = test.data.numpy()
        self.test_labels = test.targets.numpy()
        self.num_train = len(self.train_labels)
        self.num_classes = 10

    def get_nested_indices(self, fractions: List[float], seed: int) -> Dict[float, np.ndarray]:
        """Get nested stratified indices where smaller subsets are contained in larger ones."""
        indices_map = {}
        sorted_fracs = sorted(fractions)
        all_idx = np.arange(self.num_train)

        _, current = train_test_split(all_idx, test_size=sorted_fracs[0],
                                      stratify=self.train_labels, random_state=seed)
        current = np.sort(current)
        indices_map[sorted_fracs[0]] = current
        remaining = np.setdiff1d(all_idx, current)

        for i, frac in enumerate(sorted_fracs[1:], 1):
            if frac >= 1.0:
                indices_map[frac] = all_idx
                continue
            target_size = int(self.num_train * frac)
            needed = target_size - len(current)
            if needed > 0 and len(remaining) > 0:
                rem_labels = self.train_labels[remaining]
                if needed >= len(remaining):
                    additional = remaining
                else:
                    _, additional = train_test_split(remaining, test_size=needed / len(remaining),
                                                     stratify=rem_labels, random_state=seed + i)
                current = np.sort(np.concatenate([current, additional]))
                remaining = np.setdiff1d(all_idx, current)
            indices_map[frac] = current
        return indices_map

    def create_loaders(self, indices: np.ndarray, batch_size: int,
                       val_split: float, seed: int) -> Tuple[DataLoader, DataLoader]:
        """Create train/val dataloaders from indices."""
        data, labels = self.train_data[indices], self.train_labels[indices]
        train_idx, val_idx = train_test_split(np.arange(len(indices)), test_size=val_split,
                                              stratify=labels, random_state=seed)

        def make_loader(idx, shuffle):
            x = torch.tensor(data[idx], dtype=torch.float32).unsqueeze(1) / 255.0
            x = self.transform(x)
            y = torch.tensor(labels[idx], dtype=torch.long)
            return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)

        return make_loader(train_idx, True), make_loader(val_idx, False)

    def create_noisy_loaders(self, clean_indices: np.ndarray, noisy_indices: np.ndarray,
                             noise_rate: float, batch_size: int, val_split: float,
                             seed: int) -> Tuple[DataLoader, DataLoader, int]:
        """Create dataloaders with clean base + noisy additional data."""
        clean_data = self.train_data[clean_indices]
        clean_labels = self.train_labels[clean_indices]

        if len(noisy_indices) > 0:
            noisy_data = self.train_data[noisy_indices]
            noisy_labels = self.train_labels[noisy_indices].copy()

            # Apply noise
            rng = np.random.RandomState(seed)
            noise_mask = rng.random(len(noisy_labels)) < noise_rate
            num_corrupted = noise_mask.sum()
            for i in np.where(noise_mask)[0]:
                new_label = rng.randint(0, self.num_classes - 1)
                if new_label >= noisy_labels[i]:
                    new_label += 1
                noisy_labels[i] = new_label

            all_data = np.concatenate([clean_data, noisy_data])
            all_labels = np.concatenate([clean_labels, noisy_labels])
        else:
            all_data, all_labels = clean_data, clean_labels
            num_corrupted = 0

        train_idx, val_idx = train_test_split(np.arange(len(all_labels)), test_size=val_split,
                                              stratify=all_labels, random_state=seed)

        def make_loader(idx, shuffle):
            x = torch.tensor(all_data[idx], dtype=torch.float32).unsqueeze(1) / 255.0
            x = self.transform(x)
            y = torch.tensor(all_labels[idx], dtype=torch.long)
            return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)

        return make_loader(train_idx, True), make_loader(val_idx, False), int(num_corrupted)

    def get_test_loader(self, batch_size: int) -> DataLoader:
        """Create test dataloader."""
        x = torch.tensor(self.test_data, dtype=torch.float32).unsqueeze(1) / 255.0
        x = self.transform(x)
        y = torch.tensor(self.test_labels, dtype=torch.long)
        return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=False)