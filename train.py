import torch
from torch.utils.data import DataLoader
from torch import nn
from data_prep import Video3DClips
from model3d import Simple3DCNN

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_ds = Video3DClips("data/train", clip_len=16, size=112, stride=2, training=True)
    val_ds   = Video3DClips("data/val",   clip_len=16, size=112, stride=2, training=False)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=2, shuffle=False, num_workers=2, pin_memory=True)

    model = Simple3DCNN(num_classes=2).to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_val = 0.0

    for epoch in range(1, 6):  # 2 epochs for sanity
        model.train()
        total, correct, loss_sum = 0, 0, 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = torch.tensor(y, device=device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)

        train_acc = correct / total
        train_loss = loss_sum / total

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
        print(f"Epoch {epoch} | train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), "best_model.pt")
            print("trained_model.pt")

if __name__ == "__main__":
    main()

