"""Classification layer: category, urgency and sentiment, plus the rule-based baseline.

Category  : TF-IDF (word 1-3 grams + char 2-5 grams) -> Logistic Regression (probabilities for confidence)
Urgency   : same TF-IDF + numeric cues (days pending, urgent-cue count, "!" count) -> Logistic Regression
Sentiment : TF-IDF -> Logistic Regression
Low-confidence complaints are flagged for human review (ethics requirement in the case study).
"""
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from nlp_pipeline import Pipeline, SpellCorrector, URGENT_CUES

CATEGORIES = ["Street Lighting", "Water Supply", "Roads and Potholes", "Waste Collection"]
REVIEW_THRESHOLD = 0.55   # below this confidence a human officer checks the complaint

# ---- rule-based baseline ----------------------------------------------------------------
RULES = {
    "Street Lighting": {"street", "light", "lamp", "deepa", "kambada", "kattale", "bulb", "pole", "dark",
                        "uriyuttilla", "lights", "flicker", "fuse", "fused", "lite"},
    "Water Supply": {"water", "neeru", "tap", "pipe", "pipeline", "supply", "tanker", "borewell", "kaveri",
                     "cauvery", "soru", "leak", "leaking", "drinking", "kudiyuva", "barthilla", "pressure"},
    "Roads and Potholes": {"pothole", "gundi", "raste", "road", "footpath", "asphalt", "tar", "manhole",
                           "speed", "breaker", "gravel", "tarred", "repair", "highway"},
    "Waste Collection": {"garbage", "kasa", "waste", "dustbin", "trash", "dump", "vilebari", "truck", "dabba",
                         "bin", "debris", "rubbish", "tegiyalla", "tegiyiri", "smoke"},
}


def rule_based_category(pipeline, text):
    toks = set(pipeline.feature_text(text).split())
    # stems differ from the surface forms, so compare against stemmed lexicons as well
    from nltk.stem import SnowballStemmer
    st = SnowballStemmer("english")
    scores = {}
    for cat, words in RULES.items():
        stems = {st.stem(w) if w.isascii() else w for w in words}
        scores[cat] = len(toks & (words | stems))
    best = max(scores.values())
    if best == 0:
        return CATEGORIES[0]
    return next(c for c in CATEGORIES if scores[c] == best)


# ---- ML classifier ----------------------------------------------------------------------
def _vectorizer():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, min_df=1)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
    ])


class GrievanceClassifier:
    def __init__(self, c_cat=10.0, c_urg=5.0, c_sent=5.0):
        self.pipeline = Pipeline()
        self.vec = _vectorizer()
        self.cat = LogisticRegression(C=c_cat, max_iter=2000)
        self.urg = LogisticRegression(C=c_urg, max_iter=2000)
        self.sent = LogisticRegression(C=c_sent, max_iter=2000)

    # numeric urgency cues, scaled to ~[0, 1]
    def _numeric(self, texts):
        rows = []
        for t in texts:
            a = self.pipeline.analyse(t)
            rows.append([min(a["duration_days"], 14) / 14.0,
                         min(len(a["urgent_cues"]), 4) / 4.0,
                         min(t.count("!"), 3) / 3.0])
        return csr_matrix(np.array(rows))

    def fit_spell(self, texts):
        toks = [self.pipeline.normalise_tokens(t, correct=False) for t in texts]
        self.pipeline.spell = SpellCorrector().fit(toks, min_count=4)

    def features(self, texts, fit=False):
        clean = [self.pipeline.feature_text(t) for t in texts]
        X = self.vec.fit_transform(clean) if fit else self.vec.transform(clean)
        return X, hstack([X, self._numeric(texts) * 2.0]).tocsr()

    def fit(self, texts, y_cat, y_urg, y_sent):
        texts = list(texts)
        self.fit_spell(texts)
        X, X_num = self.features(texts, fit=True)
        self.cat.fit(X, y_cat)
        self.urg.fit(X_num, y_urg)
        self.sent.fit(X, y_sent)
        return self

    def predict_all(self, texts):
        texts = list(texts)
        X, X_num = self.features(texts)
        return {
            "category": self.cat.predict(X), "category_proba": self.cat.predict_proba(X),
            "urgency": self.urg.predict(X_num), "sentiment": self.sent.predict(X),
        }

    def predict(self, text):
        out = self.predict_all([text])
        probs = out["category_proba"][0]
        i = int(np.argmax(probs))
        conf = float(probs[i])
        analysis = self.pipeline.analyse(text)
        urgency = str(out["urgency"][0])
        # safety override: explicit emergency words always escalate to High
        if {"emergency", "accident", "unsafe"} & set(analysis["normalised_tokens"]):
            urgency = "High"
        return {
            "category": str(self.cat.classes_[i]),
            "confidence": round(conf, 3),
            "probabilities": {str(c): round(float(p), 3) for c, p in zip(self.cat.classes_, probs)},
            "urgency": urgency,
            "sentiment": str(out["sentiment"][0]),
            "needs_review": conf < REVIEW_THRESHOLD,
            "analysis": analysis,
        }
