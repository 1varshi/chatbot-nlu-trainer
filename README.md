# Milestone-1: Chatbot NLU Trainer & Evaluator (Scaffold)

This is a minimal project scaffold for Milestone-1 containing:
- User registration & login (JWT stored in session)
- Project workspace creation
- Dataset upload (CSV / JSON)
- Parsing uploaded dataset to extract intents, examples, and entities
- Display parsed intents and example sentences

## How to run (local)
1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Initialize the database (first run):
   ```bash
   export FLASK_APP=app.py
   flask db init  # optional, this scaffold uses simple sqlite auto-create
   ```
4. Run the app:
   ```bash
   python app.py
   ```
5. Open http://127.0.0.1:5000 in your browser.

## Notes
- This is a scaffold for Milestone-1. It focuses on functionality, not production security.
- JWT tokens are used for API endpoints; session holds a basic login state for the UI.
- Uploaded datasets are saved under `uploads/` and parsed to show intents and example sentences.
