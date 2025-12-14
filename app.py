import os
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, send_from_directory, flash
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import json, csv
import pandas as pd
from nlu_model import analyze_text
from flask import session, redirect, url_for

ADMIN_EMAIL = "adminn@gmail.com"
ADMIN_PASSWORD = "admin123" 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'dev-key-change-this'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'app.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db = SQLAlchemy(app)
with app.app_context():
    db.create_all()

def add_log(action, user):
    logs = json.load(open("logs.json"))
    logs.append({
        "action": action,
        "user": user,
        "time": str(datetime.datetime.now())
    })
    open("logs.json", "w").write(json.dumps(logs, indent=2))
# Simple models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120))
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    projects = db.relationship('Project', backref='owner', lazy=True)

class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    datasets = db.relationship('Dataset', backref='project', lazy=True)

class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(256))
    parsed = db.Column(db.Text)  # JSON string of parsed content
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)


# Simple auth helpers (DO NOT use in production as-is)
def current_user():
    user_id = session.get('user_id')
    if not user_id:
        return None
    return User.query.get(user_id)




@app.route('/')
def index():
    user = current_user()
    return render_template('index.html', user=user)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        if User.query.filter_by(email=email).first():
            flash('Email already registered', 'danger')
            return redirect(url_for('register'))
        user = User(name=name, email=email, password=password)
        db.session.add(user)
        db.session.commit()
        session['user_id'] = user.id
        return redirect(url_for('dashboard'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        user = User.query.filter_by(email=email, password=password).first()
        if not user:
            flash('Invalid credentials', 'danger')
            return redirect(url_for('login'))
        session['user_id'] = user.id
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

@app.route('/dashboard')
def dashboard():
    user = current_user()
    if not user:
        return redirect(url_for('login'))
    projects = Project.query.filter_by(user_id=user.id).all()
    return render_template('dashboard.html', user=user, projects=projects)

@app.route('/project/create', methods=['POST'])
def create_project():
    user = current_user()
    if not user:
        return redirect(url_for('login'))
    name = request.form.get('project_name') or 'New Project'
    p = Project(name=name, owner=user)
    db.session.add(p)
    db.session.commit()
    return redirect(url_for('project_view', project_id=p.id))

@app.route('/project/<int:project_id>')
def project_view(project_id):
    user = current_user()
    if not user:
        return redirect(url_for('login'))
    project = Project.query.get_or_404(project_id)
    datasets = Dataset.query.filter_by(project_id=project.id).all()
    return render_template('project.html', project=project, datasets=datasets)

def parse_dataset_file(filepath):
    parsed = []
    name = os.path.basename(filepath)

    # CSV ONLY TEXT
    try:
        df = pd.read_csv(filepath)

        if "text" not in df.columns:
            return [{"text": "ERROR: CSV must contain a 'text' column"}]

        for _, row in df.iterrows():
            parsed.append({
                "text": str(row["text"])
            })

    except Exception as e:
        parsed.append({"text": f"Parse Error: {str(e)}"})

    return parsed


@app.route('/project/<int:project_id>/upload', methods=['GET', 'POST'])
def upload_dataset(project_id):
    user = current_user()
    if not user:
        return redirect(url_for('login'))
    project = Project.query.get_or_404(project_id)
    if request.method == 'POST':
        file = request.files.get('dataset_file')
        if not file:
            flash('No file selected', 'warning')
            return redirect(request.url)
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        parsed = parse_dataset_file(save_path)
        ds = Dataset(filename=filename, parsed=json.dumps(parsed), project=project)
        db.session.add(ds)
        db.session.commit()
        flash('File uploaded and parsed', 'success')
        return redirect(url_for('project_view', project_id=project.id))
    return render_template('upload.html', project=project)

@app.route('/dataset/<int:dataset_id>')
def view_dataset(dataset_id):
    user = current_user()
    if not user:
        return redirect(url_for('login'))
    ds = Dataset.query.get_or_404(dataset_id)
    parsed = json.loads(ds.parsed or '[]')
    return render_template('dataset.html', dataset=ds, parsed=parsed)

@app.route('/dataset/<int:dataset_id>/download')
def download_dataset(dataset_id):
    ds = Dataset.query.get_or_404(dataset_id)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], ds.filename)
    return send_from_directory(app.config['UPLOAD_FOLDER'], ds.filename, as_attachment=True)


@app.route('/dataset/<int:dataset_id>/delete', methods=['POST'])
def delete_dataset(dataset_id):
    user = current_user()
    if not user:
        return redirect(url_for('login'))

    ds = Dataset.query.get_or_404(dataset_id)
    project_id = ds.project_id

    # Delete the file from /uploads/
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], ds.filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    # Delete from database
    db.session.delete(ds)
    db.session.commit()

    return redirect(url_for('project_view', project_id=project_id))

# ---------------------- MILESTONE 2: ANNOTATION TOOL ----------------------

@app.route('/annotation-tool')
def annotation_tool():
    return render_template("annotation_interface.html")


@app.route('/analyze_text', methods=['POST'])
def analyze_text_route():
    text = request.json['text']
    result = analyze_text(text)
    return jsonify(result)


@app.route('/save_annotated_data', methods=['POST'])
def save_annotated_data():
    data = request.json  # contains sentence + intent + entities

    file_path = "annotated_dataset.json"

    # Step 1: Load existing JSON array
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                annotations = json.load(f)   # Load existing JSON array
            except:
                annotations = []             # If file is broken or empty
    else:
        annotations = []

    # Step 2: Append new annotation (this includes intent)
    annotations.append(data)

    # Step 3: Save updated JSON array back to file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=4)

    return {"status": "saved"}

@app.route('/view-annotations')
def view_annotations():
    file_path = "annotated_dataset.json"

    # Load the JSON array
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                annotations = json.load(f)
            except:
                annotations = []
    else:
        annotations = []

    return render_template("view_annotations.html", annotations=annotations)

@app.route('/train', methods=['POST'])
def train_model_route():
    import subprocess, sys
    result = subprocess.run([sys.executable, "train_model.py"], capture_output=True, text=True)
    if result.returncode == 0:
        return {"status": "success", "message": "Model trained successfully!"}
    else:
        return {"status": "error", "message": result.stderr}

        # -----------------------------
# ENTITY EXTRACTION FUNCTION
# -----------------------------
def extract_entities(sentence):
    sentence_lower = sentence.lower()
    entities = {}

    # Example location keywords
    locations = ["delhi", "mumbai", "hyderabad", "chennai", "bangalore"]
    for loc in locations:
        if loc in sentence_lower:
            entities["location"] = loc

    # Example date keywords
    dates = ["today", "tomorrow", "next week"]
    for d in dates:
        if d in sentence_lower:
            entities["date"] = d

    # Example numbers (like ticket count)
    import re
    number = re.findall(r'\b\d+\b', sentence)
    if number:
        entities["number"] = number[0]

    return entities


@app.route('/test_model', methods=['GET', 'POST'])
def test_model_route():
    import joblib

    if request.method == "POST":
        sentence = request.form.get("sentence")

        # Load full pipeline (TFIDF + model)
        model = joblib.load("models/logreg.pkl")

        # --- PREDICT INTENT ---
        pred_intent = model.predict([sentence])[0]

        # --- CONFIDENCE SCORE ---
        try:
            proba = model.predict_proba([sentence])[0]
            confidence = round(max(proba), 3)
        except:
            confidence = "N/A"   # For models that don't support predict_proba

        # --- ENTITY EXTRACTION ---
        entities = extract_entities(sentence)

        # Return results to the test page
        return render_template(
            "test_model.html",
            prediction=pred_intent,
            confidence=confidence,
            entities=entities
        )

    # GET request → show empty page
    return render_template("test_model.html")




@app.route('/evaluation')
def evaluation_page():
    return render_template("evaluation.html")

@app.route('/run_evaluation', methods=['POST'])
def run_evaluation():
    import subprocess, sys, json, os

    # Run evaluation script
    subprocess.run([sys.executable, "evaluate_model.py"], capture_output=True, text=True)

    # Load metrics.json
    metrics_path = "evaluation/logreg_metrics.json"
    metrics = None
    if os.path.exists(metrics_path):
        metrics = open(metrics_path, "r").read()

    return render_template(
        "evaluation.html",
        metrics=metrics,
        confusion_url="/static/eval/logreg_confusion_matrix.png",
        entity_url="/static/eval/logreg_entity_comparison.png"
    )

@app.route('/compare_models')
def compare_models_page():
    comparison = session.get("comparison")
    graph = session.get("comparison_graph")

    return render_template(
        "compare_models.html",
        comparison=comparison,
        graph=graph
    )



@app.route('/run_comparison', methods=['POST'])
def run_comparison():
    import json, os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt


    model_a = request.form.get("model_a")
    model_b = request.form.get("model_b")

    # Load metrics
    def load_metrics(model):
        path = f"evaluation/{model}_metrics.json"
        if not os.path.exists(path):
            return None, None
        data = json.loads(open(path).read())
        return data["weighted avg"]["f1-score"], data["accuracy"]

    f1_a, acc_a = load_metrics(model_a)
    f1_b, acc_b = load_metrics(model_b)

    # Save results in session
    session['comparison'] = {
        "model_a": model_a,
        "model_b": model_b,
        "acc_a": acc_a,
        "acc_b": acc_b,
        "f1_a": f1_a,
        "f1_b": f1_b
    }

    # Create graph
    labels = ["Accuracy", "F1 Score"]
    a_vals = [acc_a, f1_a]
    b_vals = [acc_b, f1_b]

    plt.figure(figsize=(7, 5))
    x = range(len(labels))
    plt.bar([p - 0.2 for p in x], a_vals, label=model_a, width=0.4)
    plt.bar([p + 0.2 for p in x], b_vals, label=model_b, width=0.4)
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()

    graph_path = "static/eval/model_compare_two.png"
    plt.savefig(graph_path)
    plt.close()

    session['comparison_graph'] = "/" + graph_path

    return redirect("/compare_models")

@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        if email == ADMIN_EMAIL and password == ADMIN_PASSWORD:
            session["admin"] = True
            return redirect("/admin_dashboard")
        else:
            return "Invalid admin credentials"

    return render_template("admin_login.html")

@app.route("/admin_dashboard")
def admin_dashboard():
    if "admin" not in session or session["admin"] != True:
        return redirect("/admin_login")
    
    return render_template("admin_dashboard.html")


@app.route("/admin/users")
def admin_users():
    if "admin" not in session:
        return redirect("/admin_login")

    users = User.query.all()
    return render_template("admin_users.html", users=users)

@app.route("/admin/delete_user/<int:user_id>")
def admin_delete_user(user_id):
    if "admin" not in session:
        return redirect("/admin_login")

    user = User.query.get(user_id)
    db.session.delete(user)
    db.session.commit()
    return redirect("/admin/users")

@app.route("/admin/dataset")
def admin_dataset():
    data = json.load(open("annotated_dataset.json"))
    return render_template("admin_dataset.html", data=data)

@app.route("/admin/models")
def admin_models():
    models = os.listdir("models")
    return render_template("admin_models.html", models=models)

@app.route("/download_model/<name>")
def download_model(name):
    return send_from_directory("models", name, as_attachment=True)

@app.route("/delete_model/<name>")
def delete_model(name):
    os.remove(f"models/{name}")
    return redirect("/admin/models")

@app.route("/admin/logs")
def admin_logs():
    logs = json.load(open("logs.json"))
    return render_template("admin_logs.html", logs=logs)
@app.route("/test_admin")
def test_admin():
    return "Admin route working"


if __name__ == '__main__':
    app.run(debug=True)
