"""
Flask web application for Support Ticket Classification.
Simple web interface for users to classify support tickets.
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Global variables for model and artifacts
model = None
vocab = None
id2label = None
max_len = 200
device = torch.device("cpu")


def tokenize(text: str):
    return str(text).lower().split()


def encode(text: str, vocab: dict, max_len: int):
    unk = vocab["<UNK>"]
    pad = vocab["<PAD>"]
    toks = tokenize(text)
    ids = [vocab.get(tok, unk) for tok in toks][:max_len]
    if len(ids) < max_len:
        ids += [pad] * (max_len - len(ids))
    return ids


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
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(num_filters) for _ in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        hidden_dim = num_filters * len(kernel_sizes)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        emb = self.embedding(x)
        emb = emb.transpose(1, 2)
        pooled = []
        for conv, bn in zip(self.convs, self.batch_norms):
            c = conv(emb)
            c = bn(c)
            c = F.relu(c)
            p = torch.max(c, dim=2).values
            pooled.append(p)
        out = torch.cat(pooled, dim=1)
        out = self.dropout(out)
        out = F.relu(self.fc1(out))
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits


def load_model_and_artifacts():
    """Load model and artifacts on startup."""
    global model, vocab, id2label
    
    print("Loading model and artifacts...")
    
    # Load vocabulary
    with open("artifacts/vocab_balanced.json", "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    # Load label mappings
    with open("artifacts/label_map_balanced.json", "r", encoding="utf-8") as f:
        label_data = json.load(f)
    
    id2label = {int(k): v for k, v in label_data["id2label"].items()}
    num_classes = len(label_data["labels"])
    
    # Load model
    pad_id = vocab["<PAD>"]
    model = SimpleTextCNN(
        vocab_size=len(vocab),
        embed_dim=150,
        num_classes=num_classes,
        pad_id=pad_id,
        num_filters=128,
        dropout=0.3
    ).to(device)
    
    model.load_state_dict(torch.load("models/cnn_balanced.pt", map_location=device, weights_only=True))
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of classes: {num_classes}")


@torch.no_grad()
def predict(text: str, threshold: float = 0.75):
    """Predict department for given text."""
    encoded = encode(text, vocab, max_len)
    x = torch.tensor([encoded], dtype=torch.long).to(device)
    
    logits = model(x)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    
    pred_id = int(probs.argmax())
    pred_label = id2label[pred_id]
    confidence = float(probs[pred_id])
    
    # Determine action based on threshold
    if confidence < threshold:
        action = "ROUTE_TO_HUMAN"
        action_class = "warning"
        reason = f"Confidence {confidence:.1%} is below threshold {threshold:.1%}"
    else:
        action = "AUTO_ROUTE"
        action_class = "success"
        reason = f"Confidence {confidence:.1%} is above threshold {threshold:.1%}"
    
    # Get all probabilities sorted
    all_probs = {id2label[i]: float(probs[i]) for i in range(len(probs))}
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    
    return {
        "predicted_label": pred_label,
        "confidence": confidence,
        "action": action,
        "action_class": action_class,
        "reason": reason,
        "all_probabilities": sorted_probs
    }


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_route():
    """API endpoint for predictions."""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        threshold = float(data.get('threshold', 0.75))
        
        if not text:
            return jsonify({'error': 'Please enter some text'}), 400
        
        if not (0.0 <= threshold <= 1.0):
            return jsonify({'error': 'Threshold must be between 0 and 1'}), 400
        
        result = predict(text, threshold)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})


if __name__ == '__main__':
    # Load model on startup
    load_model_and_artifacts()
    
    # Run the app
    print("\nStarting Flask server...")
    print("Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
