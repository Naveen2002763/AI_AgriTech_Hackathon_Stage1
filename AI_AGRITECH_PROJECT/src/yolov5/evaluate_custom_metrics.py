import os
import argparse
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', type=str, required=True, help='Path to predicted labels folder')
    parser.add_argument('--gt', type=str, required=True, help='Path to ground truth labels folder')
    parser.add_argument('--names', type=str, required=True, help='Path to classes.txt')
    return parser.parse_args()

def read_labels(path):
    labels = {}
    for file in os.listdir(path):
        if file.endswith('.txt'):
            with open(os.path.join(path, file), 'r') as f:
                lines = f.readlines()
                labels[file] = [int(line.strip().split()[0]) for line in lines]
    return labels

def flatten(d):
    result = []
    for v in d.values():
        result.extend(v)
    return result

def main():
    args = parse_args()

    pred_labels = read_labels(args.pred)
    gt_labels = read_labels(args.gt)

    pred_flat = []
    gt_flat = []

    for file in gt_labels:
        gt = gt_labels.get(file, [])
        pred = pred_labels.get(file, [])
        # pad to match lengths
        max_len = max(len(gt), len(pred))
        gt += [999] * (max_len - len(gt))
        pred += [999] * (max_len - len(pred))
        gt_flat.extend(gt)
        pred_flat.extend(pred)

    # Remove padding (999)
    filtered = [(g, p) for g, p in zip(gt_flat, pred_flat) if g != 999 and p != 999]
    gt_clean = [g for g, _ in filtered]
    pred_clean = [p for _, p in filtered]

    with open(args.names, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    print("\nClassification Report:\n")
    print(classification_report(gt_clean, pred_clean, target_names=class_names))

    cm = confusion_matrix(gt_clean, pred_clean)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    out_path = Path("confusion_matrix.png")
    plt.savefig(out_path)
    print(f"\nSaved confusion matrix to: {out_path}")

if __name__ == '__main__':
    main()