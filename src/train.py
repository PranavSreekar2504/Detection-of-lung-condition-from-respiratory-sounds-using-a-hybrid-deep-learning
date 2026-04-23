import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset, WeightedRandomSampler
from src.dataset import ICBHIDataset, CoswaraDataset, get_class_weights
from src.model import HybridResNetLungDetector
from tqdm import tqdm
import numpy as np

def make_balanced_sampler(dataset):
    """
    Creates a WeightedRandomSampler that gives each class an equal chance
    of being selected in every mini-batch, regardless of class size.
    """
    # Collect all labels from ConcatDataset
    all_labels = []
    for d in dataset.datasets:
        all_labels.extend(d.labels)
    all_labels = np.array(all_labels)

    class_counts = np.bincount(all_labels, minlength=6)
    class_counts[class_counts == 0] = 1  # Avoid division by zero

    # Per-sample weight = inverse of its class frequency
    sample_weights = 1.0 / class_counts[all_labels]
    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


def train_model(data_dir, epochs=20, batch_size=16, learning_rate=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Datasets
    print("Loading datasets...")
    # Cap each ICBHI class at 100 samples — prevents COPD (793) from dominating
    icbhi_dataset = ICBHIDataset(data_dir=data_dir, is_train=True, max_samples_per_class=100)
    # Cap Coswara Normal at 100, keep all COVID (up to 100 too for balance)
    coswara_dataset = CoswaraDataset(
        data_dir='/Users/pranavsreekar/Desktop/Lung/Coswara-Data',
        is_train=True,
        max_normal_samples=100
    )

    full_dataset = ConcatDataset([icbhi_dataset, coswara_dataset])

    if len(full_dataset) == 0:
        print("Error: No valid audio files found!")
        return

    # Report class distribution
    all_labels = icbhi_dataset.labels + coswara_dataset.labels
    from collections import Counter
    class_names = ['Normal', 'Asthma', 'Pneumonia', 'COPD', 'Bronchitis', 'COVID-19']
    dist = Counter(all_labels)
    print("\nDataset Class Distribution:")
    for i, name in enumerate(class_names):
        print(f"  {name}: {dist.get(i, 0)} samples")
    print(f"  TOTAL: {len(full_dataset)} samples\n")

    # 2. Train / Val Split (80/20) — split BEFORE making the sampler
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Turn off augmentation for validation
    for d in full_dataset.datasets:
        d.is_train = False  # Will be re-enabled for train below

    # Re-enable augmentation for training split
    for d in full_dataset.datasets:
        d.is_train = True   # Enable globally; val will not augment since we won't touch it separately

    # 3. Build Balanced Sampler for training
    # We need labels from train_dataset indices only
    train_indices = train_dataset.indices
    all_full_labels = np.array(icbhi_dataset.labels + coswara_dataset.labels)
    train_labels = all_full_labels[train_indices]

    class_counts = np.bincount(train_labels, minlength=6)
    class_counts[class_counts == 0] = 1
    sample_weights = 1.0 / class_counts[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )

    # 4. Weighted loss for additional push
    class_weights = get_class_weights(full_dataset).to(device)
    print(f"Class Weights: {class_weights.cpu().numpy()}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 5. Initialize Model
    model = HybridResNetLungDetector(num_classes=6).to(device)

    # 6. Loss & Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float('inf')
    save_dir = 'backend'
    os.makedirs(save_dir, exist_ok=True)

    # 7. Training Loop
    print("Starting Training...\n")
    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_steps = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)
            train_steps += inputs.size(0)

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()

        train_loss = train_loss / train_steps
        train_acc = train_correct.float() / train_steps

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_steps = 0
        # Per-class tracking for diagnostics
        val_class_correct = np.zeros(6)
        val_class_total = np.zeros(6)

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                val_steps += inputs.size(0)

                for c in range(6):
                    mask = labels == c
                    val_class_correct[c] += (preds[mask] == labels[mask]).sum().item()
                    val_class_total[c] += mask.sum().item()

        val_loss = val_loss / val_steps
        val_acc = val_correct.float() / val_steps

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # Per-class accuracy
        for i, name in enumerate(class_names):
            if val_class_total[i] > 0:
                acc = val_class_correct[i] / val_class_total[i]
                print(f"  {name}: {acc:.2%} ({int(val_class_correct[i])}/{int(val_class_total[i])})")

        # Save Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(save_dir, 'cascade_hybrid_model.pth')
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best model (val_loss={val_loss:.4f}) to {save_path}\n")


if __name__ == '__main__':
    dataset_dir = 'data/ICBHI'
    train_model(dataset_dir, epochs=20)