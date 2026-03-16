"""Configuration for experiments."""

import torch

# Device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# Shared constants
CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

CONFIG = {
    "num_classes": 10,
    "val_split": 0.1,
    "dropout": 0.5,
    "batch_size": 64,
    "epochs": 50,
    "early_stopping_patience": 10,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "seeds": [14994796, 20194682, 31380194, 73962792, 99719928],    # determines amount of times the experiments are run
    # used seed exp1: 42, 123, 456 (see results in raw)

    "exp1": {
        "subset_percentages": list(range(5, 101, 5)),
    },

    "exp2": {
        "clean_samples": 30000,
        "noise_rates": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
        "noisy_percentages": list(range(0, 55, 5)),
    },

    "exp3": {
        "class_counts": list(range(2, 11)),         # [2, 3, 4, 5, 6, 7, 8, 9, 10]
        "subset_percentages": list(range(10, 101, 10)),  # coarser: [10, 20, 30, ..., 100]
        "num_class_samples": 5,                     # number of random class combinations per class count
        "target_accuracies": [0.90],                # target accuracies for iso-accuracy analysis
    },
}