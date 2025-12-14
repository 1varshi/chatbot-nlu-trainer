Chatbot NLU Trainer & Evaluator

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


