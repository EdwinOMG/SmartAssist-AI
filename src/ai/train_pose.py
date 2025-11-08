# src/ai/train_pose.py
import os
import argparse
import time
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from src.ai.dataset import PoseSequenceDataset
from src.ai.model_pose import PoseLSTM, PoseTCN
from sklearn.metrics import accuracy_score

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/pose_dataset", help="path to pose dataset root")
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--model", type=str, default="lstm", choices=["lstm", "tcn"])
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--save_dir", type=str, default="models/trained_models")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()

def collate_fn(batch):
    # batch: list of (seq_tensor, label_tensor, meta)
    seqs = [b[0] for b in batch]
    labels = [b[1] for b in batch]
    seqs = torch.stack(seqs, dim=0)  # (B, T, F)
    labels = torch.stack(labels, dim=0)
    return seqs, labels

def train_loop(args):
    os.makedirs(args.save_dir, exist_ok=True)
    dataset = PoseSequenceDataset(args.data, sequence_length=args.seq_len)
    num_classes = len(dataset.class_to_idx)
    print(f"Dataset size: {len(dataset)}, classes: {dataset.class_to_idx}")

    # split
    n_val = max(1, int(len(dataset) * args.val_split))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, collate_fn=collate_fn, num_workers=args.workers)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, collate_fn=collate_fn, num_workers=args.workers)

    # instantiate model (need to fetch feat_dim from a sample)
    sample_seq, _, _ = dataset[0]
    feat_dim = sample_seq.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model == "lstm":
        model = PoseLSTM(feat_dim=feat_dim, hidden_size=args.hidden, num_layers=2, num_classes=num_classes).to(device)
    else:
        model = PoseTCN(feat_dim=feat_dim, num_classes=num_classes).to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_val_acc = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    # training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        all_preds = []
        all_labels = []
        t0 = time.time()
        for i, (seqs, labels) in enumerate(train_loader):
            seqs = seqs.to(device)  # (B, T, F)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(seqs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * seqs.size(0)
            preds = logits.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())

        train_loss = epoch_loss / len(train_loader.dataset)
        train_acc = accuracy_score(all_labels, all_preds)

        # validation
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for seqs, labels in val_loader:
                seqs = seqs.to(device)
                labels = labels.to(device)
                logits = model(seqs)
                preds = logits.argmax(dim=1).cpu().numpy()
                val_preds.extend(preds.tolist())
                val_labels.extend(labels.cpu().numpy().tolist())
        val_acc = accuracy_score(val_labels, val_preds)

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  time={elapsed:.1f}s")

        # checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        ckpt_path = os.path.join(args.save_dir, f"ckpt_epoch{epoch+1}.pth")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "args": vars(args),
        }, ckpt_path)
        if is_best:
            best_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)
            print(f"Saved best model to {best_path}")

    print("Training complete. Best val acc:", best_val_acc)
    return os.path.join(args.save_dir, "best_model.pth")

if __name__ == "__main__":
    args = parse_args()
    train_loop(args)