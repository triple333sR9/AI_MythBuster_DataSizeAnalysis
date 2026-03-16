"""Training loop with early stopping."""

import time
from typing import Dict

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CONFIG


def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                device: torch.device, verbose: bool = True) -> Dict:
    """Train model with early stopping."""
    optimizer = Adam(model.parameters(), lr=CONFIG["learning_rate"],
                     weight_decay=CONFIG["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    patience = CONFIG["early_stopping_patience"]
    max_epochs = CONFIG["epochs"]

    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_val_loss, best_state, patience_counter, best_epoch = float("inf"), None, 0, 0
    start_time = time.time()

    iterator = tqdm(range(max_epochs), desc="Training") if verbose else range(max_epochs)
    for epoch in iterator:
        # Train
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                val_loss += criterion(out, y).item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += x.size(0)
        val_loss, val_acc = val_loss / total, correct / total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        if verbose:
            iterator.set_postfix(train_loss=f"{train_loss:.4f}", val_acc=f"{val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss, best_epoch, patience_counter = val_loss, epoch, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    tqdm.write(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state:
        model.load_state_dict(best_state)

    return {
        "epochs_trained": len(history["train_loss"]),
        "best_epoch": best_epoch + 1,
        "best_val_accuracy": max(history["val_accuracy"]),
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "training_time_seconds": time.time() - start_time,
        "history": history
    }