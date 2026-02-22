import torch
from torch.utils.data import DataLoader
from torch import nn
from collections import Counter
from data_prep import Video3DClips
from model3d import VideoResNet2D


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # -------------------------
    # DATASET SETTINGS
    # -------------------------
    clip_len = 8
    size = 224

    train_ds = Video3DClips(
        "data/train",
        clip_len=clip_len,
        size=size,
        stride=2,
        training=True
    )

    val_ds = Video3DClips(
        "data/val",
        clip_len=clip_len,
        size=size,
        stride=2,
        training=False
    )

    print("Training label distribution:",
          Counter([y for _, y in train_ds.samples]))

    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # -------------------------
    # MODEL
    # -------------------------
    model = VideoResNet2D(num_classes=2).to(device)

    # Freeze backbone first (very important)
    for param in model.feature_extractor.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    best_val = 0.0
    epochs = 15

    # -------------------------
    # TRAINING LOOP
    # -------------------------
    for epoch in range(1, epochs + 1):

        # Unfreeze after 5 epochs
        if epoch == 6:
            print("Unfreezing backbone...")
            for param in model.feature_extractor.parameters():
                param.requires_grad = True

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=1e-5   # lower LR for fine-tuning
            )

        model.train()
        total, correct, loss_sum = 0, 0, 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = torch.tensor(y, device=device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        train_acc = correct / total
        train_loss = loss_sum / total

        # -------------------------
        # VALIDATION
        # -------------------------
        model.eval()
        total, correct = 0, 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = torch.tensor(y, device=device)

                logits = model(x)
                pred = logits.argmax(1)

                correct += (pred == y).sum().item()
                total += x.size(0)

        val_acc = correct / total

        print(f"Epoch {epoch:02d} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.3f} | "
              f"Val Acc: {val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), "best_model.pt")
            print("✔ Saved best_model.pt")

    print("Training complete.")
    print("Best validation accuracy:", best_val)


if __name__ == "__main__":
    main()
