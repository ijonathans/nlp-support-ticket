# Customer Support Ticket Auto-Routing System

An NLP-based system that automatically routes customer support tickets to the correct department, with a confidence-based fallback to human agents.

## 🌐 Live Demo

**Try it online:** [Deployed on Railway](https://your-app.up.railway.app) *(Update after deployment)*

## 🚀 Quick Start - Local Development

**Run the web app locally:**

```bash
python app.py
```

Then open: **http://localhost:5000**

Or use the batch file (Windows):
```bash
start_webapp.bat
```

## 🔍 Problem
Manual ticket routing is slow, inconsistent, and expensive. This project builds a production-oriented text classification pipeline that balances accuracy, latency, and operational risk.

## 🧠 Approach

### Baseline Models (scikit-learn)
- TF-IDF text representation
- Multinomial Naive Bayes
- Linear SVM

### Deep Learning Model (PyTorch) - **Production Model**
- CNN-based text classifier with balanced class weights
- **Test Accuracy: 67.2%**
- **Test Macro-F1: 0.582**
- **Latency: 1.28ms/sample**
- Training: 30 epochs with sqrt-softened class weights

### Human-in-the-Loop
- Confidence thresholding for reliable predictions
- Low-confidence predictions routed to human agents
- **Recommended threshold: 0.75** (70% automation, 78% accuracy)

## 📊 Model Performance

| Department | Precision | Recall | F1-Score |
|------------|-----------|--------|----------|
| Billing | 0.91 | 0.75 | 0.82 |
| Customer Service | 0.53 | 0.56 | 0.54 |
| HR | 1.00 | 0.36 | 0.53 |
| Product Support | 0.56 | 0.51 | 0.53 |
| Returns | 0.70 | 0.38 | 0.49 |
| Sales | 0.65 | 0.27 | 0.39 |
| Tech Support | 0.71 | 0.84 | 0.77 |

**Overall Accuracy:** 67.2% | **Macro-F1:** 0.582