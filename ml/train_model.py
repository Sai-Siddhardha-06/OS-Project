#!/usr/bin/env python3
"""
ml/train_model.py
Trains 9 -> 16 -> 8 -> 4 MLP neural network using NumPy.
Features:
- Group split by blueprint_id (holding out entire blueprints for val and test)
- Normalization (mean and standard deviation computed on train set only)
- Adam optimizer with cross-entropy loss
- Comparison against baselines (Majority Class, Logistic Regression)
- Sim-to-Real evaluation on real xv6 traces (ml/real_traces.csv)
- Exports trained float weights and metrics
"""

import json
import csv
import os
import numpy as np

np.random.seed(42)

def load_dataset(csv_path):
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = []
        for r in reader:
            data.append([float(x) for x in r])
    data = np.array(data)
    blueprint_ids = data[:, 0].astype(int)
    y = data[:, 1].astype(int)
    X = data[:, 2:]
    return blueprint_ids, X, y, header[2:]

def load_real_traces(csv_path):
    if not os.path.exists(csv_path):
        return None, None
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        data = []
        for r in reader:
            data.append([float(x) for x in r[1:]])
    data = np.array(data)
    y_real = data[:, 0].astype(int)
    X_real = data[:, 1:]
    return X_real, y_real

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    # Numerically stable softmax
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

class MLP:
    def __init__(self, in_dim=9, h1_dim=16, h2_dim=8, out_dim=4):
        self.in_dim = in_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.out_dim = out_dim
        
        # He initialization
        self.W1 = np.random.randn(in_dim, h1_dim) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros(h1_dim)
        self.W2 = np.random.randn(h1_dim, h2_dim) * np.sqrt(2.0 / h1_dim)
        self.b2 = np.zeros(h2_dim)
        self.W3 = np.random.randn(h2_dim, out_dim) * np.sqrt(2.0 / h2_dim)
        self.b3 = np.zeros(out_dim)
        
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = relu(self.z2)
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.probs = softmax(self.z3)
        return self.probs

    def backward(self, X, y_onehot):
        m = X.shape[0]
        # Cross-entropy with softmax derivative: probs - y
        dz3 = (self.probs - y_onehot) / m
        dW3 = np.dot(self.a2.T, dz3)
        db3 = np.sum(dz3, axis=0)
        
        da2 = np.dot(dz3, self.W3.T)
        dz2 = da2 * (self.z2 > 0)
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0)
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * (self.z1 > 0)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0)
        
        return [dW1, db1, dW2, db2, dW3, db3]

class LogisticRegression:
    def __init__(self, in_dim=9, out_dim=4):
        self.W = np.zeros((in_dim, out_dim))
        self.b = np.zeros(out_dim)
        
    def forward(self, X):
        z = np.dot(X, self.W) + self.b
        return softmax(z)
        
    def train(self, X, y, lr=0.05, epochs=300):
        y_onehot = np.eye(4)[y]
        m = X.shape[0]
        for _ in range(epochs):
            probs = self.forward(X)
            dz = (probs - y_onehot) / m
            dW = np.dot(X.T, dz)
            db = np.sum(dz, axis=0)
            self.W -= lr * dW
            self.b -= lr * db

def evaluate_predictions(y_true, y_pred, probs=None):
    acc = np.mean(y_true == y_pred)
    classes = [0, 1, 2, 3]
    cm = np.zeros((4, 4), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
        
    per_class = {}
    for c in classes:
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        fn = np.sum(cm[c, :]) - tp
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[c] = {"precision": float(prec), "recall": float(rec), "f1": float(f1), "support": int(np.sum(cm[c, :]))}
        
    return {
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class
    }

def main():
    dataset_path = "/home/sai/OS Project/ml/dataset.csv"
    real_traces_path = "/home/sai/OS Project/ml/real_traces.csv"
    metrics_path = "/home/sai/OS Project/ml/model_metrics.json"
    weights_path = "/home/sai/OS Project/ml/model_weights.json"
    
    blueprint_ids, X_raw, y, feat_names = load_dataset(dataset_path)
    
    # Group split by blueprint_id:
    # 20 blueprints (5 per class: Interactive 0-4, IO 10-14, CPU 20-24, Mixed 30-34)
    # Train: 3 per class (IDs 0,1,2, 10,11,12, 20,21,22, 30,31,32)
    # Val: 1 per class (IDs 3, 13, 23, 33)
    # Test: 1 per class (IDs 4, 14, 24, 34)
    train_bps = {0, 1, 2, 10, 11, 12, 20, 21, 22, 30, 31, 32}
    val_bps   = {3, 13, 23, 33}
    test_bps  = {4, 14, 24, 34}
    
    train_mask = np.isin(blueprint_ids, list(train_bps))
    val_mask   = np.isin(blueprint_ids, list(val_bps))
    test_mask  = np.isin(blueprint_ids, list(test_bps))
    
    X_train_raw, y_train = X_raw[train_mask], y[train_mask]
    X_val_raw,   y_val   = X_raw[val_mask],   y[val_mask]
    X_test_raw,  y_test  = X_raw[test_mask],  y[test_mask]
    
    # Compute normalization on train set only
    mean = np.mean(X_train_raw, axis=0)
    std = np.std(X_train_raw, axis=0)
    std[std == 0] = 1.0 # avoid divide by zero
    
    X_train = (X_train_raw - mean) / std
    X_val   = (X_val_raw - mean) / std
    X_test  = (X_test_raw - mean) / std
    
    # Baselines
    # 1. Majority class
    majority_class = int(np.bincount(y_train).argmax())
    maj_pred = np.full_like(y_test, majority_class)
    maj_acc = float(np.mean(y_test == maj_pred))
    
    # 2. Logistic Regression
    lr_model = LogisticRegression(in_dim=9, out_dim=4)
    lr_model.train(X_train, y_train, lr=0.1, epochs=400)
    lr_pred = np.argmax(lr_model.forward(X_test), axis=1)
    lr_metrics = evaluate_predictions(y_test, lr_pred)
    
    # Train MLP
    mlp = MLP(in_dim=9, h1_dim=16, h2_dim=8, out_dim=4)
    y_train_onehot = np.eye(4)[y_train]
    
    # Adam parameters
    lr = 0.005
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    params = [mlp.W1, mlp.b1, mlp.W2, mlp.b2, mlp.W3, mlp.b3]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    
    epochs = 400
    batch_size = 64
    best_val_loss = float('inf')
    best_params = None
    
    num_batches = int(np.ceil(X_train.shape[0] / batch_size))
    
    for epoch in range(1, epochs + 1):
        # Shuffle
        perm = np.random.permutation(X_train.shape[0])
        X_shuff = X_train[perm]
        y_shuff = y_train_onehot[perm]
        
        for b in range(num_batches):
            start = b * batch_size
            end = min(X_train.shape[0], start + batch_size)
            xb = X_shuff[start:end]
            yb = y_shuff[start:end]
            
            mlp.forward(xb)
            grads = mlp.backward(xb, yb)
            
            # Adam step
            for i in range(len(params)):
                m[i] = beta1 * m[i] + (1 - beta1) * grads[i]
                v[i] = beta2 * v[i] + (1 - beta2) * (grads[i] ** 2)
                m_hat = m[i] / (1 - beta1 ** epoch)
                v_hat = v[i] / (1 - beta2 ** epoch)
                params[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)
                
        # Validation loss
        val_probs = mlp.forward(X_val)
        val_loss = -np.mean(np.log(val_probs[np.arange(len(y_val)), y_val] + 1e-12))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = [p.copy() for p in params]
            
    # Restore best parameters
    for i in range(len(params)):
        params[i][:] = best_params[i]
        
    # Evaluate on held-out test blueprints
    test_probs = mlp.forward(X_test)
    test_preds = np.argmax(test_probs, axis=1)
    mlp_test_metrics = evaluate_predictions(y_test, test_preds, test_probs)
    
    # Sim-to-Real Evaluation on real xv6 traces
    X_real_raw, y_real = load_real_traces(real_traces_path)
    real_metrics = None
    if X_real_raw is not None and len(X_real_raw) > 0:
        X_real_norm = (X_real_raw - mean) / std
        real_probs = mlp.forward(X_real_norm)
        real_preds = np.argmax(real_probs, axis=1)
        real_metrics = evaluate_predictions(y_real, real_preds, real_probs)
        print(f"Sim-to-Real Evaluation on {len(X_real_raw)} real xv6 traces: Accuracy = {real_metrics['accuracy']:.4f}")
        
    print(f"Held-out Blueprint Test Set: Accuracy = {mlp_test_metrics['accuracy']:.4f}")
    print(f"Logistic Regression Baseline: Accuracy = {lr_metrics['accuracy']:.4f}")
    print(f"Majority Class Baseline: Accuracy = {maj_acc:.4f}")
    
    # Save metrics
    metrics_data = {
        "architecture": "9 -> 16 -> 8 -> 4",
        "normalization": {
            "mean": mean.tolist(),
            "std": std.tolist()
        },
        "majority_class_baseline_accuracy": maj_acc,
        "logistic_regression_baseline": lr_metrics,
        "mlp_held_out_test_metrics": mlp_test_metrics,
        "sim_to_real_metrics": real_metrics
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
        
    # Save float weights
    weights_data = {
        "W1": mlp.W1.tolist(),
        "b1": mlp.b1.tolist(),
        "W2": mlp.W2.tolist(),
        "b2": mlp.b2.tolist(),
        "W3": mlp.W3.tolist(),
        "b3": mlp.b3.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist()
    }
    with open(weights_path, "w") as f:
        json.dump(weights_data, f, indent=2)
        
    print(f"Model saved to {weights_path} and metrics to {metrics_path}")

if __name__ == "__main__":
    main()
