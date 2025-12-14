# compare_models.py
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

EVAL_DIR = Path("evaluation")
STATIC_EVAL = Path("static/eval")

models = ["logreg", "svm", "nb"]
scores = {}

for m in models:
    metrics_file = EVAL_DIR / f"{m}_metrics.json"
    if metrics_file.exists():
        data = json.loads(metrics_file.read_text())
        f1 = data["weighted avg"]["f1-score"]
        acc = data["accuracy"]
        scores[m] = {"accuracy": acc, "f1": f1}

# Plot comparison (F1-score)
plt.figure(figsize=(7,5))
plt.title("Model Comparison (F1 Score)")
plt.bar(scores.keys(), [scores[m]["f1"] for m in scores])
plt.ylabel("F1 Score")
plt.savefig("static/eval/model_comparison.png")
plt.close()

print("Model comparison generated!")

