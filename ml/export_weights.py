#!/usr/bin/env python3
"""
ml/export_weights.py
Converts trained float MLP weights to Q16.16 fixed-point C header (kernel/model_weights.h).
Generates:
- Normalization constants (mean, std in Q16.16)
- Layer weights and biases (W1, b1, W2, b2, W3, b3 in Q16.16)
- Fixed-point exp lookup table for softmax
- Class-to-Queue mapping table (Table[4][3])
- Test vectors (20+ samples) with reference outputs for kernel self-test
- Evaluates float vs fixed-point accuracy and agreement
"""

import json
import csv
import os
import numpy as np

Q_SHIFT = 16
Q_ONE = 1 << Q_SHIFT  # 65536

def float_to_q16(val):
    if isinstance(val, (list, np.ndarray)):
        return np.round(np.array(val) * Q_ONE).astype(np.int32)
    return int(round(val * Q_ONE))

# Fixed-point Exp Lookup Table: 256 entries for range [-8.0, 0.0]
EXP_TABLE_ENTRIES = 256
EXP_MAX_NEG = 8.0  # inputs below -8.0 map to 0

def make_exp_table():
    table = []
    for i in range(EXP_TABLE_ENTRIES):
        # i = 0 corresponds to delta = 0.0, i = 255 corresponds to delta = -8.0
        neg_val = (i / (EXP_TABLE_ENTRIES - 1)) * EXP_MAX_NEG
        exp_val = np.exp(-neg_val)
        table.append(int(round(exp_val * Q_ONE)))
    return table

EXP_TABLE = make_exp_table()

def fixed_point_exp(delta_q16):
    """
    delta_q16 <= 0 in Q16.16.
    Returns exp(delta) in Q16.16.
    """
    if delta_q16 >= 0:
        return Q_ONE
    # -8.0 in Q16.16 is -8 * 65536 = -524288
    min_neg = -int(EXP_MAX_NEG * Q_ONE)
    if delta_q16 <= min_neg:
        return 0
    # Map [-min_neg, 0] to [255, 0]
    # idx = (-delta_q16 * (EXP_TABLE_ENTRIES - 1)) / (EXP_MAX_NEG * 65536)
    idx = int((-delta_q16 * (EXP_TABLE_ENTRIES - 1)) // (int(EXP_MAX_NEG) << Q_SHIFT))
    if idx >= EXP_TABLE_ENTRIES:
        idx = EXP_TABLE_ENTRIES - 1
    return EXP_TABLE[idx]

def fixed_point_forward(feats, mean_q16, std_q16, W1_q16, b1_q16, W2_q16, b2_q16, W3_q16, b3_q16):
    """
    Bit-exact fixed point forward pass using integer arithmetic.
    feats: list/array of 9 raw integers.
    Returns:
      probs_permille: array of 4 integers (0..1000 summing to 1000)
      pred_class: argmax (0..3)
      pred_conf: confidence in permille (0..1000)
    """
    # 1. Normalization: x_norm = ((raw - mean) << 16) / std
    x_norm = np.zeros(9, dtype=np.int32)
    for i in range(9):
        diff = (int(feats[i]) << Q_SHIFT) - int(mean_q16[i])
        x_norm[i] = int((diff << Q_SHIFT) // int(std_q16[i]))

    # 2. Layer 1: 9 -> 16 + ReLU
    z1 = np.zeros(16, dtype=np.int32)
    for j in range(16):
        acc = int(b1_q16[j])
        for i in range(9):
            acc += (int(x_norm[i]) * int(W1_q16[i, j])) >> Q_SHIFT
        z1[j] = max(0, acc)  # ReLU

    # 3. Layer 2: 16 -> 8 + ReLU
    z2 = np.zeros(8, dtype=np.int32)
    for j in range(8):
        acc = int(b2_q16[j])
        for i in range(16):
            acc += (int(z1[i]) * int(W2_q16[i, j])) >> Q_SHIFT
        z2[j] = max(0, acc)  # ReLU

    # 4. Layer 3: 8 -> 4 (logits)
    z3 = np.zeros(4, dtype=np.int32)
    for j in range(4):
        acc = int(b3_q16[j])
        for i in range(8):
            acc += (int(z2[i]) * int(W3_q16[i, j])) >> Q_SHIFT
        z3[j] = acc

    # 5. Softmax via Exp Table
    max_z = np.max(z3)
    exp_vals = np.zeros(4, dtype=np.int64)
    for j in range(4):
        delta = z3[j] - max_z  # <= 0
        exp_vals[j] = fixed_point_exp(delta)

    sum_exp = np.sum(exp_vals)
    if sum_exp == 0:
        sum_exp = 1

    probs_permille = np.zeros(4, dtype=np.int32)
    for j in range(4):
        probs_permille[j] = int((exp_vals[j] * 1000) // sum_exp)

    # Ensure sum is exactly 1000
    cur_sum = np.sum(probs_permille)
    if cur_sum != 1000:
        probs_permille[np.argmax(probs_permille)] += (1000 - cur_sum)

    pred_class = int(np.argmax(probs_permille))
    pred_conf = int(probs_permille[pred_class])

    return probs_permille, pred_class, pred_conf

def main():
    weights_path = "/home/sai/OS Project/ml/model_weights.json"
    header_path = "/home/sai/OS Project/xv6-riscv/kernel/model_weights.h"
    dataset_path = "/home/sai/OS Project/ml/dataset.csv"
    real_traces_path = "/home/sai/OS Project/ml/real_traces.csv"

    with open(weights_path) as f:
        w = json.load(f)

    mean_float = np.array(w["mean"])
    std_float = np.array(w["std"])
    W1_float = np.array(w["W1"])
    b1_float = np.array(w["b1"])
    W2_float = np.array(w["W2"])
    b2_float = np.array(w["b2"])
    W3_float = np.array(w["W3"])
    b3_float = np.array(w["b3"])

    # Quantize to Q16.16
    mean_q16 = float_to_q16(mean_float)
    std_q16 = float_to_q16(std_float)
    W1_q16 = float_to_q16(W1_float)
    b1_q16 = float_to_q16(b1_float)
    W2_q16 = float_to_q16(W2_float)
    b2_q16 = float_to_q16(b2_float)
    W3_q16 = float_to_q16(W3_float)
    b3_q16 = float_to_q16(b3_float)

    # Validate Float vs Fixed-point on dataset test set
    with open(dataset_path) as f:
        rows = list(csv.reader(f))[1:]
    test_rows = [r for r in rows if int(r[0]) in {4, 14, 24, 34}]

    float_preds = []
    fixed_preds = []
    y_true = []

    for r in test_rows:
        y = int(r[1])
        feats = [int(float(x)) for x in r[2:]]
        y_true.append(y)

        # Float forward
        x_norm_fl = (np.array(feats) - mean_float) / std_float
        z1_fl = np.maximum(0, np.dot(x_norm_fl, W1_float) + b1_float)
        z2_fl = np.maximum(0, np.dot(z1_fl, W2_float) + b2_float)
        z3_fl = np.dot(z2_fl, W3_float) + b3_float
        ez = np.exp(z3_fl - np.max(z3_fl))
        fl_probs = ez / np.sum(ez)
        float_preds.append(int(np.argmax(fl_probs)))

        # Fixed forward
        _, fx_pred, _ = fixed_point_forward(feats, mean_q16, std_q16, W1_q16, b1_q16, W2_q16, b2_q16, W3_q16, b3_q16)
        fixed_preds.append(fx_pred)

    float_acc = np.mean(np.array(float_preds) == np.array(y_true))
    fixed_acc = np.mean(np.array(fixed_preds) == np.array(y_true))
    agreement = np.mean(np.array(float_preds) == np.array(fixed_preds))

    print(f"Test Set Float Accuracy:       {float_acc:.4f}")
    print(f"Test Set Fixed-Point Accuracy: {fixed_acc:.4f}")
    print(f"Float vs Fixed-Point Agreement: {agreement:.4f} ({(agreement*100):.2f}%)")

    # Pick 24 diverse test vectors for kernel self-test:
    # 4 from real traces (one per class), plus 20 from synthetic blueprints
    test_vectors = []
    # Real traces first
    with open(real_traces_path) as f:
        real_r = list(csv.reader(f))[1:]
    # one of each class
    for target_c in [0, 1, 2, 3]:
        for row in real_r:
            if int(row[1]) == target_c:
                feats = [int(float(x)) for x in row[2:]]
                probs, pred, conf = fixed_point_forward(feats, mean_q16, std_q16, W1_q16, b1_q16, W2_q16, b2_q16, W3_q16, b3_q16)
                test_vectors.append({
                    "name": f"real_{row[0]}",
                    "true_class": target_c,
                    "feats": feats,
                    "expected_probs": probs.tolist(),
                    "expected_pred": pred,
                    "expected_conf": conf
                })
                break

    # Add 20 synthetic vectors (5 from each class across different blueprints)
    for c in [0, 1, 2, 3]:
        matching = [r for r in test_rows if int(r[1]) == c][:5]
        for idx, r in enumerate(matching):
            feats = [int(float(x)) for x in r[2:]]
            probs, pred, conf = fixed_point_forward(feats, mean_q16, std_q16, W1_q16, b1_q16, W2_q16, b2_q16, W3_q16, b3_q16)
            test_vectors.append({
                "name": f"synth_c{c}_bp{r[0]}_{idx}",
                "true_class": c,
                "feats": feats,
                "expected_probs": probs.tolist(),
                "expected_pred": pred,
                "expected_conf": conf
            })

    print(f"Generated {len(test_vectors)} test vectors for kernel self-test.")

    # Class to queue mapping table (in permille 0..1000)
    # Interactive (0): Q0=850, Q1=100, Q2=50
    # IO-bound (1):    Q0=700, Q1=250, Q2=50
    # CPU-bound (2):   Q0=50,  Q1=150, Q2=800
    # Mixed (3):       Q0=150, Q1=700, Q2=150
    class_queue_table = [
        [850, 100, 50],
        [700, 250, 50],
        [50,  150, 800],
        [150, 700, 150]
    ]

    # Prior marginal queue distribution for Mode 4 (RAND-CTRL)
    prior_queue_dist = [333, 334, 333]

    # Format C header
    with open(header_path, "w") as f:
        f.write("/*\n")
        f.write(" * kernel/model_weights.h\n")
        f.write(" * Auto-generated by ml/export_weights.py. DO NOT HAND-EDIT.\n")
        f.write(f" * Float Accuracy: {float_acc:.4f}, Fixed Accuracy: {fixed_acc:.4f}, Agreement: {agreement:.4f}\n")
        f.write(" */\n\n")
        f.write("#ifndef MODEL_WEIGHTS_H\n")
        f.write("#define MODEL_WEIGHTS_H\n\n")
        f.write('#include "types.h"\n\n')
        f.write("#define HAS_TRAINED_WEIGHTS 1\n")
        f.write("#define Q_SHIFT 16\n")
        f.write("#define Q_ONE (1 << Q_SHIFT)\n")
        f.write("#define NUM_FEATURES 9\n")
        f.write("#define H1_DIM 16\n")
        f.write("#define H2_DIM 8\n")
        f.write("#define OUT_DIM 4\n")
        f.write("#define NUM_QUEUES 3\n\n")

        # Normalization
        f.write("static const int32 model_mean[NUM_FEATURES] = {\n  ")
        f.write(", ".join(str(int(x)) for x in mean_q16))
        f.write("\n};\n\n")

        f.write("static const int32 model_std[NUM_FEATURES] = {\n  ")
        f.write(", ".join(str(int(x)) for x in std_q16))
        f.write("\n};\n\n")

        # W1, b1
        f.write("static const int32 model_W1[NUM_FEATURES][H1_DIM] = {\n")
        for i in range(9):
            f.write("  { " + ", ".join(f"{int(x)}" for x in W1_q16[i]) + " },\n")
        f.write("};\n\n")

        f.write("static const int32 model_b1[H1_DIM] = {\n  ")
        f.write(", ".join(str(int(x)) for x in b1_q16))
        f.write("\n};\n\n")

        # W2, b2
        f.write("static const int32 model_W2[H1_DIM][H2_DIM] = {\n")
        for i in range(16):
            f.write("  { " + ", ".join(f"{int(x)}" for x in W2_q16[i]) + " },\n")
        f.write("};\n\n")

        f.write("static const int32 model_b2[H2_DIM] = {\n  ")
        f.write(", ".join(str(int(x)) for x in b2_q16))
        f.write("\n};\n\n")

        # W3, b3
        f.write("static const int32 model_W3[H2_DIM][OUT_DIM] = {\n")
        for i in range(8):
            f.write("  { " + ", ".join(f"{int(x)}" for x in W3_q16[i]) + " },\n")
        f.write("};\n\n")

        f.write("static const int32 model_b3[OUT_DIM] = {\n  ")
        f.write(", ".join(str(int(x)) for x in b3_q16))
        f.write("\n};\n\n")

        # Exp Table
        f.write(f"#define EXP_TABLE_ENTRIES {EXP_TABLE_ENTRIES}\n")
        f.write("#define EXP_MAX_NEG 8\n")
        f.write("static const int32 exp_table[EXP_TABLE_ENTRIES] = {\n")
        for i in range(0, EXP_TABLE_ENTRIES, 8):
            chunk = EXP_TABLE[i:i+8]
            f.write("  " + ", ".join(str(x) for x in chunk) + ",\n")
        f.write("};\n\n")

        # Queue mapping tables
        f.write("static const int32 class_queue_prob[OUT_DIM][NUM_QUEUES] = {\n")
        for r in class_queue_table:
            f.write("  { " + ", ".join(str(x) for x in r) + " },\n")
        f.write("};\n\n")

        f.write("static const int32 rand_ctrl_queue_prob[NUM_QUEUES] = {\n  ")
        f.write(", ".join(str(x) for x in prior_queue_dist))
        f.write("\n};\n\n")

        # Test vectors for self-test
        f.write(f"#define NUM_SELFTEST_VECTORS {len(test_vectors)}\n")
        f.write("struct ml_test_vector {\n")
        f.write("  const char *name;\n")
        f.write("  int32 feats[NUM_FEATURES];\n")
        f.write("  int32 expected_probs[OUT_DIM];\n")
        f.write("  int expected_pred;\n")
        f.write("  int32 expected_conf;\n")
        f.write("};\n\n")

        f.write("static const struct ml_test_vector selftest_vectors[NUM_SELFTEST_VECTORS] = {\n")
        for tv in test_vectors:
            feat_s = ", ".join(str(x) for x in tv["feats"])
            prob_s = ", ".join(str(x) for x in tv["expected_probs"])
            f.write(f'  {{ "{tv["name"]}", {{ {feat_s} }}, {{ {prob_s} }}, {tv["expected_pred"]}, {tv["expected_conf"]} }},\n')
        f.write("};\n\n")

        f.write("#endif /* MODEL_WEIGHTS_H */\n")

    print(f"Exported model weights and test vectors to {header_path}")

if __name__ == "__main__":
    main()
