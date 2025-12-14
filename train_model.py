# train_model.py
import json, joblib, os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).parent
DATA_FILE = ROOT / "annotated_dataset.json"   # adapt if your filename differs
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

def load_data():
    raw = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    texts = [item["sentence"] for item in raw]
    labels = [item["intent"] for item in raw]
    return texts, labels, raw

def train():
    X, y, raw = load_data()
    X_train, X_test, y_train, y_test, train_raw, test_raw = train_test_split(
        X, y, raw, test_size=0.2, random_state=42

    )
    # vectorize + model pipelines
    pipelines = {
        "logreg": make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000)),
        "svm": make_pipeline(TfidfVectorizer(), LinearSVC()),
        "nb": make_pipeline(TfidfVectorizer(), MultinomialNB()),
    }
    for name, pipe in pipelines.items():
        print("Training", name)
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, MODELS_DIR / f"{name}.pkl")
    # save test split for evaluation
    (MODELS_DIR / "test_split.json").write_text(json.dumps(test_raw, indent=2), encoding="utf-8")
    print("Training complete. Models saved to", MODELS_DIR)

if __name__ == "__main__":
    train()
