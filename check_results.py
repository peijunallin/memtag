import os
import csv
from collections import defaultdict

base_dir = [REDACTED]
target_suffix = "topk50_soft3.csv"
keywords = ""

# Dictionary to store scores: file -> list of (dataset, f1, em)
all_scores = defaultdict(list)

# Dictionary to store dataset-wise scores: dataset -> list of (file, f1, em)
dataset_scores = defaultdict(list)

# Traverse directory tree
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(target_suffix):
            full_path = os.path.join(root, file)
            with open(full_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dataset = row['dataset']
                    f1 = float(row['f1_score'])
                    em = float(row['exact_match'])
                    all_scores[file].append((dataset, f1, em))
                    dataset_scores[dataset].append((file, f1, em))

# Compute average scores and sort
averaged_scores = []
for file, entries in all_scores.items():
    avg_f1 = sum(f1 for _, f1, _ in entries) / len(entries)
    avg_em = sum(em for _, _, em in entries) / len(entries)
    avg_total = (avg_f1 + avg_em) / 2
    averaged_scores.append((file, avg_f1, avg_em, avg_total, entries))

top_3 = sorted(averaged_scores, key=lambda x: x[3], reverse=True)[:3]

# Print top 3 overall models
print("Top 3 models by average (F1 + EM)/2:")
for file, avg_f1, avg_em, avg_total, entries in top_3:
    print(f"\n{file}")
    print(f"  Avg F1: {avg_f1:.4f}, Avg EM: {avg_em:.4f}, Combined: {avg_total:.4f}")
    print("  Per-dataset scores:")
    for dataset, f1, em in entries:
        print(f"    {dataset}: F1 = {f1:.4f}, EM = {em:.4f}")

# Print top 3 scores per dataset
print("\nTop 3 models per dataset by (F1 + EM)/2:")
for dataset, entries in dataset_scores.items():
    top_entries = sorted(entries, key=lambda x: (x[1] + x[2]) / 2, reverse=True)[:3]
    print(f"\nDataset: {dataset}")
    for file, f1, em in top_entries:
        combined = (f1 + em) / 2
        print(f"  {file}: F1 = {f1:.4f}, EM = {em:.4f}, Combined = {combined:.4f}")
