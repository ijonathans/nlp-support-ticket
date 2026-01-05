import os
import sys
import json
import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from training script to avoid code duplication
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("deep_models", os.path.join(os.path.dirname(__file__), "03_deep_models.py"))
    deep_models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deep_models)
    SimpleTextCNN = deep_models.SimpleTextCNN
    tokenize = deep_models.tokenize
    encode = deep_models.encode
except Exception as e:
    print(f"Warning: Could not import from 03_deep_models.py: {e}")
    print("Using fallback definitions...")
    
    # Fallback: define locally if import fails
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


def load_artifacts(artifacts_dir):
    """Load vocabulary and label mappings from artifacts directory."""
    vocab_path = os.path.join(artifacts_dir, "vocab_simple.json")
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    
    labelmap_path = os.path.join(artifacts_dir, "label_map.json")
    with open(labelmap_path, "r", encoding="utf-8") as f:
        label_data = json.load(f)
    
    id2label = {int(k): v for k, v in label_data["id2label"].items()}
    labels = label_data["labels"]
    
    return vocab, id2label, labels


def load_model(models_dir, vocab, num_classes, device):
    """Load trained CNN model from disk."""
    pad_id = vocab["<PAD>"]
    
    model = SimpleTextCNN(
        vocab_size=len(vocab),
        embed_dim=150,
        num_classes=num_classes,
        pad_id=pad_id,
        num_filters=128,
        dropout=0.3
    ).to(device)
    
    model_path = os.path.join(models_dir, "cnn_simple.pt")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    return model


@torch.no_grad()
def predict(text: str, model, vocab, id2label, max_len, device, threshold=0.0):
    """
    Predict department for a given ticket text.
    
    Args:
        text: Input ticket text
        model: Trained CNN model
        vocab: Vocabulary dictionary
        id2label: ID to label mapping
        max_len: Maximum sequence length
        device: torch device
        threshold: Confidence threshold for auto-routing
    
    Returns:
        Dictionary with prediction results
    """
    encoded = encode(text, vocab, max_len)
    x = torch.tensor([encoded], dtype=torch.long).to(device)
    
    logits = model(x)
    probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    
    pred_id = int(probs.argmax())
    pred_label = id2label[pred_id]
    confidence = float(probs[pred_id])
    
    if confidence < threshold:
        return {
            "predicted_label": pred_label,
            "confidence": confidence,
            "all_probabilities": {id2label[i]: float(probs[i]) for i in range(len(probs))},
            "action": "ROUTE_TO_HUMAN",
            "reason": f"Confidence {confidence:.2%} below threshold {threshold:.2%}"
        }
    else:
        return {
            "predicted_label": pred_label,
            "confidence": confidence,
            "all_probabilities": {id2label[i]: float(probs[i]) for i in range(len(probs))},
            "action": "AUTO_ROUTE",
            "reason": f"Confidence {confidence:.2%} above threshold {threshold:.2%}"
        }


def interactive_mode(model, vocab, id2label, max_len, device, threshold):
    """Interactive mode for continuous ticket classification."""
    print("\n" + "="*60)
    print("Support Ticket Auto-Routing System")
    print("="*60)
    print(f"\nConfidence Threshold: {threshold:.2%}")
    print("Available departments:", ", ".join(sorted(set(id2label.values()))))
    print("\nType 'quit' or 'exit' to stop\n")
    
    while True:
        print("-" * 60)
        text = input("\nEnter ticket text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if not text:
            print("WARNING: Please enter some text.")
            continue
        
        result = predict(text, model, vocab, id2label, max_len, device, threshold)
        
        print(f"\n[Prediction Results]")
        print(f"   Department: {result['predicted_label']}")
        print(f"   Confidence: {result['confidence']:.2%}")
        print(f"   Action: {result['action']}")
        print(f"   Reason: {result['reason']}")
        
        print(f"\n[All Department Probabilities]")
        sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
        for dept, prob in sorted_probs:
            bar = "#" * int(prob * 50)
            print(f"   {dept:20s} {prob:6.2%} {bar}")


def main():
    parser = argparse.ArgumentParser(
        description="Support Ticket Classification Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python 05_user_inference.py
  
  # Single prediction
  python 05_user_inference.py --text "My credit card was charged twice"
  
  # Custom threshold
  python 05_user_inference.py --threshold 0.80 --text "I need help with my order"
        """
    )
    
    parser.add_argument("--text", type=str, default=None, 
                        help="Single ticket text to classify (optional)")
    parser.add_argument("--threshold", type=float, default=0.75, 
                        help="Confidence threshold for auto-routing (default: 0.75)")
    parser.add_argument("--max_len", type=int, default=200, 
                        help="Maximum sequence length (default: 200)")
    parser.add_argument("--artifacts_dir", type=str, default="../artifacts",
                        help="Path to artifacts directory")
    parser.add_argument("--models_dir", type=str, default="../models",
                        help="Path to models directory")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load artifacts
    print("Loading model and artifacts...")
    vocab, id2label, labels = load_artifacts(args.artifacts_dir)
    model = load_model(args.models_dir, vocab, len(labels), device)
    print("[OK] Model loaded successfully!")
    print(f"   Vocabulary size: {len(vocab)}")
    print(f"   Number of classes: {len(labels)}")
    
    # Single prediction or interactive mode
    if args.text:
        result = predict(args.text, model, vocab, id2label, args.max_len, device, args.threshold)
        print("\n" + "="*60)
        print("[Prediction Results]")
        print("="*60)
        print(f"Input Text: {args.text}")
        print(f"\nPredicted Department: {result['predicted_label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Action: {result['action']}")
        print(f"Reason: {result['reason']}")
        
        print(f"\n[All Department Probabilities]")
        sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
        for dept, prob in sorted_probs:
            bar = "#" * int(prob * 50)
            print(f"   {dept:20s} {prob:6.2%} {bar}")
    else:
        interactive_mode(model, vocab, id2label, args.max_len, device, args.threshold)


if __name__ == "__main__":
    main()
