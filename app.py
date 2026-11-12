"""Flask app: citizen form, REST API and officer dashboard.

Run locally: python app.py
Production: gunicorn app:app
"""
import os
import pickle
import sqlite3
from datetime import datetime, timezone

from flask import Flask, g, jsonify, redirect, render_template, request, url_for

from classifier import GrievanceClassifier  # noqa: F401 - required to unpickle the model
from response_generator import generate_reply, make_ticket_id

BASE = os.path.dirname(os.path.abspath(__file__))
# Render's persistent disk should be mounted at /var/data. Locally, keep the DB
# alongside the source. DB_PATH can override either location.
DATA_DIR = os.environ.get("DATA_DIR", "/var/data" if os.path.isdir("/var/data") else BASE)
os.makedirs(DATA_DIR, exist_ok=True)
DB = os.environ.get("DB_PATH", os.path.join(DATA_DIR, "complaints.db"))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(BASE, "models", "model.pkl"))
MAX_LEN = 1000

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024


def load_model():
    """Load the trained classifier once when the WSGI worker imports the app."""
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. "
            "Ensure models/model.pkl is committed to the repository."
        )
    with open(MODEL_PATH, "rb") as model_file:
        return pickle.load(model_file)


MODEL = load_model()


def db():
    if "db" not in g:
        # A short timeout helps when multiple Gunicorn workers access SQLite.
        g.db = sqlite3.connect(DB, timeout=30)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_error):
    connection = g.pop("db", None)
    if connection is not None:
        connection.close()


def init_db():
    """Create the table and add newer columns to older databases safely."""
    os.makedirs(os.path.dirname(os.path.abspath(DB)), exist_ok=True)
    with sqlite3.connect(DB, timeout=30) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS complaints(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticket TEXT,
                created TEXT,
                text TEXT,
                category TEXT,
                confidence REAL,
                urgency TEXT,
                sentiment TEXT,
                location TEXT,
                time_ref TEXT,
                language TEXT,
                needs_review INTEGER,
                reply TEXT,
                department TEXT,
                status TEXT DEFAULT 'Open'
            )"""
        )
        existing = {
            row[1] for row in connection.execute("PRAGMA table_info(complaints)")
        }
        # Upgrade databases created by earlier versions without deleting records.
        if "status" not in existing:
            connection.execute(
                "ALTER TABLE complaints ADD COLUMN status TEXT DEFAULT 'Open'"
            )
        connection.commit()


def process(complaint_text):
    complaint_text = (complaint_text or "").strip()
    if not complaint_text:
        raise ValueError("Complaint text is empty.")
    if len(complaint_text) > MAX_LEN:
        raise ValueError(f"Complaint is too long (max {MAX_LEN} characters).")

    result = MODEL.predict(complaint_text)
    ticket = make_ticket_id(complaint_text, result["category"])
    reply = generate_reply(result, ticket)
    entities = result["analysis"]["entities"]

    db().execute(
        """INSERT INTO complaints(
            ticket, created, text, category, confidence, urgency, sentiment,
            location, time_ref, language, needs_review, reply, department
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            ticket,
            datetime.now(timezone.utc).isoformat(timespec="minutes"),
            complaint_text,
            result["category"],
            result["confidence"],
            result["urgency"],
            result["sentiment"],
            entities.get("location"),
            entities.get("time"),
            result["analysis"]["language_style"],
            int(result["needs_review"]),
            reply["text"],
            reply["department"],
        ),
    )
    db().commit()
    return ticket, result, reply


@app.get("/health")
def health():
    """Render health-check endpoint."""
    return jsonify(status="ok"), 200


@app.route("/", methods=["GET", "POST"])
def index():
    result = error = None
    complaint_text = ""
    if request.method == "POST":
        complaint_text = request.form.get("text", "")
        try:
            ticket, classification, reply = process(complaint_text)
            result = {"ticket": ticket, "res": classification, "reply": reply}
        except ValueError as exc:
            error = str(exc)
    return render_template(
        "index.html", result=result, error=error, text=complaint_text
    )


@app.get("/dashboard")
def dashboard():
    category = request.args.get("category", "")
    urgency = request.args.get("urgency", "")
    query = "SELECT * FROM complaints WHERE 1=1"
    args = []
    if category:
        query += " AND category=?"
        args.append(category)
    if urgency:
        query += " AND urgency=?"
        args.append(urgency)
    query += (
        " ORDER BY CASE urgency WHEN 'High' THEN 0 "
        "WHEN 'Medium' THEN 1 ELSE 2 END, id DESC LIMIT 200"
    )
    rows = db().execute(query, args).fetchall()
    return render_template(
        "dashboard.html",
        rows=rows,
        stats=_stats(),
        cat=category,
        urg=urgency,
    )


@app.post("/resolve/<int:cid>")
def resolve(cid):
    db().execute(
        "UPDATE complaints SET status='Resolved' WHERE id=?", (cid,)
    )
    db().commit()
    return redirect(request.referrer or url_for("dashboard"))


def _stats():
    connection = db()

    def rows_as_dicts(sql):
        return [dict(row) for row in connection.execute(sql).fetchall()]

    return {
        "total": connection.execute(
            "SELECT COUNT(*) FROM complaints"
        ).fetchone()[0],
        "open": connection.execute(
            "SELECT COUNT(*) FROM complaints WHERE status='Open'"
        ).fetchone()[0],
        "review": connection.execute(
            "SELECT COUNT(*) FROM complaints "
            "WHERE needs_review=1 AND status='Open'"
        ).fetchone()[0],
        "by_category": rows_as_dicts(
            "SELECT category k, COUNT(*) n FROM complaints "
            "GROUP BY category ORDER BY n DESC"
        ),
        "by_urgency": rows_as_dicts(
            "SELECT urgency k, COUNT(*) n FROM complaints "
            "GROUP BY urgency"
        ),
        "by_location": rows_as_dicts(
            "SELECT location k, COUNT(*) n FROM complaints "
            "WHERE location IS NOT NULL GROUP BY location "
            "ORDER BY n DESC LIMIT 8"
        ),
    }


@app.route("/api/complaints", methods=["GET", "POST"])
def api_complaints():
    if request.method == "POST":
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(error="Send a JSON object containing 'text'."), 400
        try:
            ticket, result, reply = process(data.get("text"))
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
        analysis = result["analysis"]
        return (
            jsonify(
                ticket=ticket,
                category=result["category"],
                confidence=result["confidence"],
                probabilities=result["probabilities"],
                urgency=result["urgency"],
                sentiment=result["sentiment"],
                needs_review=result["needs_review"],
                entities=analysis["entities"],
                language=analysis["language_style"],
                senses=analysis["senses"],
                reply=reply["text"],
                department=reply["department"],
            ),
            201,
        )

    rows = db().execute(
        "SELECT * FROM complaints ORDER BY id DESC LIMIT 100"
    ).fetchall()
    return jsonify([dict(row) for row in rows])


@app.get("/api/stats")
def api_stats():
    return jsonify(_stats())


# Safe for Gunicorn import and local development.
init_db()

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False,
    )
