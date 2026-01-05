import os
import json
import time
import random
import argparse
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenize(text: str):
    return str(text).lower().split()


def build_vocab(texts, max_vocab=20000, min_freq=2):
    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))

    vocab = {"<PAD>": 0, "<UNK>": 1}
    for word, freq in counter.most_common():
        if freq < min_freq:
            continue
        if word not in vocab:
            vocab[word] = len(vocab)
        if len(vocab) >= max_vocab:
            break
    return vocab


def encode(text: str, vocab: dict, max_len: int):
    unk = vocab["<UNK>"]
    pad = vocab["<PAD>"]
    toks = tokenize(text)
    ids = [vocab.get(tok, unk) for tok in toks][:max_len]
    if len(ids) < max_len:
        ids += [pad] * (max_len - len(ids))
    return ids


def pick_max_len(train_texts, percentile=95, min_len=50, max_cap=300):
    lengths = np.array([len(tokenize(t)) for t in train_texts])
    p = int(np.percentile(lengths, percentile))
    return max(min_len, min(p, max_cap)), lengths


def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# ---------------------------
# Dataset
# ---------------------------
class TicketDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = torch.tensor(encode(self.texts[idx], self.vocab, self.max_len), dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ---------------------------
# Model
# ---------------------------
class SimpleTextCNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_classes: int,
        pad_id: int,
        num_filters: int = 128,
        kernel_sizes=(3, 4, 5),
        dropout: float = 0.3
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        
        # Convolutional layers with batch normalization
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(num_filters) for _ in kernel_sizes])
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layers with additional hidden layer
        hidden_dim = num_filters * len(kernel_sizes)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        # x: (B, L)
        emb = self.embedding(x)          # (B, L, E)
        emb = emb.transpose(1, 2)        # (B, E, L)
        
        pooled = []
        for conv, bn in zip(self.convs, self.batch_norms):
            c = conv(emb)               # (B, F, L-k+1)
            c = bn(c)                   # Batch normalization
            c = F.relu(c)               # Activation
            p = torch.max(c, dim=2).values  # global max pool -> (B, F)
            pooled.append(p)
        
        out = torch.cat(pooled, dim=1)  # (B, F * num_kernels)
        out = self.dropout(out)
        out = F.relu(self.fc1(out))     # Hidden layer with ReLU
        out = self.dropout(out)
        logits = self.fc2(out)          # Output layer
        return logits


# ---------------------------
# Train / Eval
# ---------------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds, all_true = [], []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_true.append(yb.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_true, all_preds)
    mf1 = f1_score(all_true, all_preds, average="macro")
    return avg_loss, acc, mf1


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_true = [], []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu().numpy())
        all_true.append(yb.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_true, all_preds)
    mf1 = f1_score(all_true, all_preds, average="macro")
    return avg_loss, acc, mf1, all_true, all_preds


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    probs_list = []
    y_list = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        probs_list.append(probs)
        y_list.append(yb.numpy())
    return np.vstack(probs_list), np.concatenate(y_list)


def threshold_table(probs, y_true, thresholds):
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)

    rows = []
    for t in thresholds:
        auto = conf >= t
        coverage = float(auto.mean())
        reject = 1.0 - coverage

        if auto.sum() == 0:
            rows.append({
                "threshold": t, "coverage": coverage, "reject_rate": reject,
                "accuracy_auto": np.nan, "macro_f1_auto": np.nan, "n_auto": 0
            })
            continue

        y_auto = y_true[auto]
        p_auto = pred[auto]
        rows.append({
            "threshold": t,
            "coverage": coverage,
            "reject_rate": reject,
            "accuracy_auto": float(accuracy_score(y_auto, p_auto)),
            "macro_f1_auto": float(f1_score(y_auto, p_auto, average="macro")),
            "n_auto": int(auto.sum())
        })
    return pd.DataFrame(rows)


@torch.no_grad()
def benchmark_latency(model, batch_tensor, device, n_runs=300):
    model.eval()
    batch_tensor = batch_tensor.to(device)
    _ = model(batch_tensor)  # warmup
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = model(batch_tensor)
    t1 = time.perf_counter()
    return (t1 - t0) / n_runs * 1000.0


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=15)  # Moderate epoch count
    parser.add_argument("--lr", type=float, default=1.5e-3)  # Slightly higher LR for better learning

    parser.add_argument("--out_models_dir", type=str, default="../models")
    parser.add_argument("--out_artifacts_dir", type=str, default="../artifacts")
    parser.add_argument("--out_reports_dir", type=str, default="../reports")

    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ensure_dirs(args.out_models_dir, args.out_artifacts_dir, args.out_reports_dir)

    # 1) Load
    df = pd.read_csv("../dataset/processed/df_english_cleaned.csv")

    labels = sorted(df["label"].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    df["label_id"] = df["label"].map(label2id)

    # 3) Split
    X = df['text_clean'].values
    y = df["label_id"].values

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=42, stratify=y_tmp
    )
    print("Split sizes:", len(X_train), len(X_val), len(X_test))

    # Compute class weights to handle imbalance (using square root for softer weighting)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = np.sqrt(class_weights)  # Soften the weights to avoid over-penalizing
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("\nClass weights (softened with sqrt for better balance):")
    for i, label in enumerate(labels):
        print(f"  {label:20s}: {class_weights[i]:.3f}")

    # 4) Vocab + max_len
    vocab = build_vocab(X_train, max_vocab=25000, min_freq=2)  # Balanced vocab size
    max_len, lengths = pick_max_len(X_train, percentile=95)  # Standard percentile
    print("Vocab size:", len(vocab))
    print("Max len:", max_len, "(p95 tokens)")
    print("Mean length:", lengths.mean(), "Median:", np.median(lengths))

    # 5) Datasets/loaders
    train_ds = TicketDataset(X_train, y_train, vocab, max_len)
    val_ds = TicketDataset(X_val, y_val, vocab, max_len)
    test_ds = TicketDataset(X_test, y_test, vocab, max_len)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)  # Standard batch size
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # 6) Model
    pad_id = vocab["<PAD>"]
    num_classes = len(labels)
    model = SimpleTextCNN(
        vocab_size=len(vocab),
        embed_dim=200,  # Moderate increase from 128
        num_classes=num_classes,
        pad_id=pad_id,
        num_filters=256,  # Keep at 128 for balance
        dropout=0.3  # Reduced from 0.35 for better learning
    ).to(device)

    # Use class weights to handle imbalance (prioritize minority classes like Billing)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)  # Light weight decay
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # 7) Train with early stopping
    best_val_mf1 = -1.0
    best_state = None
    bad = 0
    history = []

    print("\n" + "="*60)
    print("Training CNN with Class-Weighted Loss")
    print("="*60)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_mf1 = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc, va_mf1, _, _ = eval_one_epoch(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc, "train_macro_f1": tr_mf1,
            "val_loss": va_loss, "val_acc": va_acc, "val_macro_f1": va_mf1
        })

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.3f} mf1 {tr_mf1:.3f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.3f} mf1 {va_mf1:.3f}"
        )

        # Step the scheduler based on validation F1
        scheduler.step(va_mf1)

        if va_mf1 > best_val_mf1:
            best_val_mf1 = va_mf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 5: #Patience - increased to 5 for better convergence with slower LR
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    print("Best VAL macro-F1:", best_val_mf1)

    # 8) VAL metrics + report
    _, _, _, y_true_val, y_pred_val = eval_one_epoch(model, val_loader, criterion, device)
    print("\nVAL report:")
    print(classification_report(y_true_val, y_pred_val, target_names=[id2label[i] for i in range(num_classes)]))

    # 9) Thresholding on VAL
    val_probs, val_true = predict_probs(model, val_loader, device)
    thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85]
    thr_df = threshold_table(val_probs, val_true, thresholds)
    print("\nThresholding (VAL):")
    print(thr_df)

    # 10) Latency
    x1, _ = val_ds[0]
    x1 = x1.unsqueeze(0)  # (1, L)
    xb, _ = next(iter(val_loader))
    xb32 = xb[:32]

    ms_single = benchmark_latency(model, x1, device, n_runs=300)
    ms_batch32 = benchmark_latency(model, xb32, device, n_runs=300)
    print(f"\nLatency: {ms_single:.3f} ms / sample | {ms_batch32:.3f} ms / batch(32)")

    # 11) TEST metrics + report
    test_probs, test_true = predict_probs(model, test_loader, device)
    test_pred = test_probs.argmax(axis=1)
    test_acc = accuracy_score(test_true, test_pred)
    test_mf1 = f1_score(test_true, test_pred, average="macro")

    print("\nTEST accuracy:", test_acc)
    print("TEST macro-F1:", test_mf1)
    print("\nTEST report:")
    print(classification_report(test_true, test_pred, target_names=[id2label[i] for i in range(num_classes)]))

    # 12) Save artifacts
    model_path = os.path.join(args.out_models_dir, "cnn_balanced.pt")
    torch.save(model.state_dict(), model_path)

    vocab_path = os.path.join(args.out_artifacts_dir, "vocab_balanced.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f)

    labelmap_path = os.path.join(args.out_artifacts_dir, "label_map_balanced.json")
    with open(labelmap_path, "w", encoding="utf-8") as f:
        json.dump({"labels": labels, "label2id": label2id, "id2label": id2label}, f, indent=2)

    hist_path = os.path.join(args.out_reports_dir, "cnn_balanced_history.csv")
    pd.DataFrame(history).to_csv(hist_path, index=False)

    thr_path = os.path.join(args.out_reports_dir, "cnn_balanced_thresholding_val.csv")
    thr_df.to_csv(thr_path, index=False)

    test_metrics_path = os.path.join(args.out_reports_dir, "cnn_balanced_test_metrics.json")
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "accuracy": float(test_acc),
                "macro_f1": float(test_mf1),
                "latency_ms_per_sample": float(ms_single),
                "latency_ms_per_batch32": float(ms_batch32),
            },
            f,
            indent=2
        )

    print("\n" + "="*60)
    print("Saved Successfully!")
    print("="*60)
    print("Model:", model_path)
    print("Vocab:", vocab_path)
    print("Label maps:", labelmap_path)
    print("History:", hist_path)
    print("Thresholding:", thr_path)
    print("Test metrics:", test_metrics_path)
    print("\nTo use this model, update app.py to load 'cnn_balanced.pt'")


if __name__ == "__main__":
    main()
