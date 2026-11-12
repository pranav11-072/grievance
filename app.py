"""Flask app: citizen form, REST API and officer dashboard.

Run:  python app.py        (opens on http://127.0.0.1:5000)
API:  POST /api/complaints   {"text": "..."}  ->  classification + generated reply (stored in SQLite)
      GET  /api/complaints   list;  GET /api/stats  aggregate counts
"""
import os
import pickle
import sqlite3
from datetime import datetime

from flask import Flask, g, jsonify, redirect, render_template, request, url_for

from classifier import GrievanceClassifier  # noqa: F401  (needed to unpickle the model)
from response_generator import generate_reply, make_ticket_id

BASE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(BASE, "complaints.db")
MAX_LEN = 1000

app = Flask(__name__)
with open(os.path.join(BASE, "models", "model.pkl"), "rb") as f:
    MODEL = pickle.load(f)


def db():
    if "db" not in g:
        g.db = sqlite3.connect(DB)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_):
    d = g.pop("db", None)
    if d:
        d.close()


def init_db():
    with sqlite3.connect(DB) as c:
        c.execute("""CREATE TABLE IF NOT EXISTS complaints(
            id INTEGER PRIMARY KEY AUTOINCREMENT, ticket TEXT, created TEXT, text TEXT, category TEXT,
            confidence REAL, urgency TEXT, sentiment TEXT, location TEXT, time_ref TEXT, language TEXT,
            needs_review INTEGER, reply TEXT, department TEXT, status TEXT DEFAULT 'Open')""")


def process(text):
    text = (text or "").strip()
    if not text:
        raise ValueError("Complaint text is empty.")
    if len(text) > MAX_LEN:
        raise ValueError(f"Complaint is too long (max {MAX_LEN} characters).")
    res = MODEL.predict(text)
    ticket = make_ticket_id(text, res["category"])
    reply = generate_reply(res, ticket)
    ent = res["analysis"]["entities"]
    db().execute(
        "INSERT INTO complaints(ticket,created,text,category,confidence,urgency,sentiment,location,time_ref,"
        "language,needs_review,reply,department) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (ticket, datetime.now().strftime("%Y-%m-%d %H:%M"), text, res["category"], res["confidence"],
         res["urgency"], res["sentiment"], ent["location"], ent["time"], res["analysis"]["language_style"],
         int(res["needs_review"]), reply["text"], reply["department"]))
    db().commit()
    return ticket, res, reply


# ----------------------------------------------------------------------------- pages
@app.route("/", methods=["GET", "POST"])
def index():
    result = error = None
    text = ""
    if request.method == "POST":
        text = request.form.get("text", "")
        try:
            ticket, res, reply = process(text)
            result = {"ticket": ticket, "res": res, "reply": reply}
        except ValueError as e:
            error = str(e)
    return render_template("index.html", result=result, error=error, text=text)


@app.route("/dashboard")
def dashboard():
    cat = request.args.get("category", "")
    urg = request.args.get("urgency", "")
    q = "SELECT * FROM complaints WHERE 1=1"
    args = []
    if cat:
        q += " AND category=?"; args.append(cat)
    if urg:
        q += " AND urgency=?"; args.append(urg)
    q += " ORDER BY CASE urgency WHEN 'High' THEN 0 WHEN 'Medium' THEN 1 ELSE 2 END, id DESC LIMIT 200"
    rows = db().execute(q, args).fetchall()
    stats = _stats()
    return render_template("dashboard.html", rows=rows, stats=stats, cat=cat, urg=urg)


@app.route("/resolve/<int:cid>", methods=["POST"])
def resolve(cid):
    db().execute("UPDATE complaints SET status='Resolved' WHERE id=?", (cid,))
    db().commit()
    return redirect(request.referrer or url_for("dashboard"))


# ----------------------------------------------------------------------------- API
def _stats():
    d = db()
    one = lambda sql: [dict(r) for r in d.execute(sql).fetchall()]
    return {
        "total": d.execute("SELECT COUNT(*) FROM complaints").fetchone()[0],
        "open": d.execute("SELECT COUNT(*) FROM complaints WHERE status='Open'").fetchone()[0],
        "review": d.execute("SELECT COUNT(*) FROM complaints WHERE needs_review=1 AND status='Open'").fetchone()[0],
        "by_category": one("SELECT category k, COUNT(*) n FROM complaints GROUP BY category ORDER BY n DESC"),
        "by_urgency": one("SELECT urgency k, COUNT(*) n FROM complaints GROUP BY urgency"),
        "by_location": one("SELECT location k, COUNT(*) n FROM complaints WHERE location IS NOT NULL "
                           "GROUP BY location ORDER BY n DESC LIMIT 8"),
    }


@app.route("/api/complaints", methods=["POST"])
def api_post():
    data = request.get_json(silent=True) or {}
    try:
        ticket, res, reply = process(data.get("text"))
    except ValueError as e:
        return jsonify(error=str(e)), 400
    a = res["analysis"]
    return jsonify(ticket=ticket, category=res["category"], confidence=res["confidence"],
                   probabilities=res["probabilities"], urgency=res["urgency"], sentiment=res["sentiment"],
                   needs_review=res["needs_review"], entities=a["entities"], language=a["language_style"],
                   senses=a["senses"], reply=reply["text"], department=reply["department"]), 201


@app.route("/api/complaints", methods=["GET"])
def api_list():
    rows = db().execute("SELECT * FROM complaints ORDER BY id DESC LIMIT 100").fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/stats")
def api_stats():
    return jsonify(_stats())


init_db()

if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get("PORT", 5000)))
