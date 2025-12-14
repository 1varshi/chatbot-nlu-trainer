# evaluate_model.py

import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

# Paths
ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
EVAL_DIR = ROOT / "evaluation"
STATIC_EVAL = ROOT / "static" / "eval"

EVAL_DIR.mkdir(exist_ok=True)
STATIC_EVAL.mkdir(parents=True, exist_ok=True)


# -------------------------------------------
# Load test data (created during training)
# -------------------------------------------
def load_test_data():
    test_file = MODELS_DIR / "test_split.json"
    if not test_file.exists():
        raise FileNotFoundError("❌ test_split.json not found. Train the model first!")
    
    return json.loads(test_file.read_text(encoding="utf-8"))


# -------------------------------------------
# Convert mismatched keys (your dataset uses different names)
# -------------------------------------------
def get_text(item):
    return item.get("text") or item.get("user_input") or item.get("query") or item.get("sentence")


def get_label(item):
    return item.get("intent") or item.get("label")


# -------------------------------------------
# Evaluate a single model
# -------------------------------------------
def evaluate_model(model_name="logreg"):
    model_path = MODELS_DIR / f"{model_name}.pkl"
    model = joblib.load(model_path)

    test_data = load_test_data()

    y_true = []
    y_pred = []

    for item in test_data:
        text = get_text(item)
        true_label = get_label(item)
        pred = model.predict([text])[0]

        y_true.append(true_label)
        y_pred.append(pred)

    # Metrics
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    # Save metrics.json
    metrics_path = EVAL_DIR / f"{model_name}_metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # -------------------------------------------
    # CONFUSION MATRIX PNG
    # -------------------------------------------
    labels = sorted(list(set(y_true)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title(f"{model_name.upper()} Confusion Matrix")
    plt.colorbar()
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(np.arange(len(labels)), labels)
    plt.tight_layout()

    cm_png = EVAL_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(cm_png)
    plt.close()

    # Copy to static folder for browser
    (STATIC_EVAL / cm_png.name).write_bytes(cm_png.read_bytes())

    # -------------------------------------------
    # ENTITY COMPARISON PNG (dummy comparison)
    # -------------------------------------------
    true_count = {}
    pred_count = {}

    for item, pred in zip(test_data, y_pred):
        ents = item.get("entities", [])
        for e in ents:
            true_count[e["label"]] = true_count.get(e["label"], 0) + 1

    # We do not predict entities → predicted = 0
    pred_count = {label: 0 for label in true_count}

    labels_e = list(true_count.keys())
    true_vals = [true_count[l] for l in labels_e]
    pred_vals = [pred_count[l] for l in labels_e]

    plt.figure(figsize=(7, 5))
    x = np.arange(len(labels_e))
    width = 0.35

    plt.bar(x - width/2, true_vals, width, label="True")
    plt.bar(x + width/2, pred_vals, width, label="Predicted")
    plt.xticks(x, labels_e, rotation=45)
    plt.title("Entity Comparison")
    plt.legend()
    plt.tight_layout()

    ent_png = EVAL_DIR / f"{model_name}_entity_comparison.png"
    plt.savefig(ent_png)
    plt.close()

    (STATIC_EVAL / ent_png.name).write_bytes(ent_png.read_bytes())

    return {
        "metrics": report,
        "confusion_matrix": cm_png.name,
        "entity_comparison": ent_png.name,
    }


# -------------------------------------------
# MAIN EXECUTION
# -------------------------------------------
if __name__ == "__main__":
    print("Running evaluation for all models...\n")

    results = {}
    for mn in ["logreg", "svm", "nb"]:
        model_path = MODELS_DIR / f"{mn}.pkl"
        if model_path.exists():
            print("Evaluating", mn)
            results[mn] = evaluate_model(mn)

    print("\nEvaluation Completed!")
    print("Metrics saved in:", EVAL_DIR)
    print("Images saved in:", STATIC_EVAL)
