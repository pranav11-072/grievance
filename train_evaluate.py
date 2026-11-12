"""Train and evaluate the grievance classifiers.

Experiments
  A. Model comparison on a stratified 80/20 split (rule-based vs TF-IDF + NB / SVM / LogReg)
  B. Generalisation check on a hand-written test set that shares no templates with training
  C. 5-fold cross-validation of the final model, per-language-style breakdown, confidence analysis

Outputs: models/model.pkl, reports/*.png, reports/metrics.json, reports/classification_report.txt
Run:  python train_evaluate.py
"""
import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from classifier import CATEGORIES, GrievanceClassifier, REVIEW_THRESHOLD, rule_based_category

os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)
SEED = 7

df = pd.read_csv("data/complaints.csv").fillna("")
hand = pd.read_csv("data/handwritten_test.csv")
train, test = train_test_split(df, test_size=0.2, stratify=df.category, random_state=SEED)
metrics = {"n_total": len(df), "n_train": len(train), "n_test": len(test), "n_handwritten": len(hand)}


def macro(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return {"accuracy": round(accuracy_score(y, p), 4), "precision": round(pr, 4),
            "recall": round(rc, 4), "f1": round(f1, 4)}


# ---------------------------------------------------------------- A. model comparison
base = GrievanceClassifier()
base.fit_spell(train.text)
Xtr_txt = [base.pipeline.feature_text(t) for t in train.text]
Xte_txt = [base.pipeline.feature_text(t) for t in test.text]
Xhw_txt = [base.pipeline.feature_text(t) for t in hand.text]


def word_vec():
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)),
    ])


models = {
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Linear SVM": LinearSVC(C=1.0),
    "Logistic Regression": LogisticRegression(C=10, max_iter=2000),
}
comparison = {}
rb_test = [rule_based_category(base.pipeline, t) for t in test.text]
rb_hand = [rule_based_category(base.pipeline, t) for t in hand.text]
comparison["Rule-based keywords"] = {"split": macro(test.category, rb_test), "handwritten": macro(hand.category, rb_hand)}
for name, m in models.items():
    v = word_vec()
    Xtr = v.fit_transform(Xtr_txt)
    m.fit(Xtr, train.category)
    comparison[name] = {
        "split": macro(test.category, m.predict(v.transform(Xte_txt))),
        "handwritten": macro(hand.category, m.predict(v.transform(Xhw_txt))),
    }
metrics["category_model_comparison"] = comparison

# ---------------------------------------------------------------- final model (LogReg, all 3 heads)
clf = GrievanceClassifier().fit(train.text, train.category, train.urgency, train.sentiment)
pt = clf.predict_all(test.text)
ph = clf.predict_all(hand.text)
metrics["final_split"] = {"category": macro(test.category, pt["category"]),
                          "urgency": macro(test.urgency, pt["urgency"]),
                          "sentiment": macro(test.sentiment, pt["sentiment"])}
metrics["final_handwritten"] = {"category": macro(hand.category, ph["category"]),
                                "urgency": macro(hand.urgency, ph["urgency"]),
                                "sentiment": macro(hand.sentiment, ph["sentiment"])}

# per-language-style breakdown on split
by_style = {}
for s in sorted(test.lang_style.unique()):
    mask = (test.lang_style == s).values
    by_style[s] = {"n": int(mask.sum()), "accuracy": round(accuracy_score(test.category[mask], pt["category"][mask]), 4)}
metrics["category_accuracy_by_style"] = by_style

# confidence analysis on handwritten set (the honest one)
conf_hw = ph["category_proba"].max(axis=1)
correct_hw = ph["category"] == hand.category.values
flag = conf_hw < REVIEW_THRESHOLD
metrics["human_review"] = {
    "threshold": REVIEW_THRESHOLD,
    "handwritten_flagged": int(flag.sum()),
    "errors_caught_by_flag": int((flag & ~correct_hw).sum()),
    "total_errors": int((~correct_hw).sum()),
    "accuracy_on_auto_handled": round(float(correct_hw[~flag].mean()), 4) if (~flag).any() else None,
}

# ---------------------------------------------------------------- C. cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv = []
for tr_i, te_i in skf.split(df.text, df.category):
    c = GrievanceClassifier().fit(df.text.iloc[tr_i], df.category.iloc[tr_i], df.urgency.iloc[tr_i], df.sentiment.iloc[tr_i])
    p = c.predict_all(df.text.iloc[te_i])
    cv.append(f1_score(df.category.iloc[te_i], p["category"], average="macro"))
metrics["cv5_category_macro_f1"] = {"mean": round(float(np.mean(cv)), 4), "std": round(float(np.std(cv)), 4)}

# ---------------------------------------------------------------- reports
with open("reports/classification_report.txt", "w") as f:
    f.write("=== CATEGORY, stratified split ===\n")
    f.write(classification_report(test.category, pt["category"], digits=3))
    f.write("\n=== CATEGORY, hand-written test set ===\n")
    f.write(classification_report(hand.category, ph["category"], digits=3))
    f.write("\n=== URGENCY, stratified split ===\n")
    f.write(classification_report(test.urgency, pt["urgency"], digits=3))
    f.write("\n=== SENTIMENT, stratified split ===\n")
    f.write(classification_report(test.sentiment, pt["sentiment"], digits=3))


def plot_cm(y, p, title, path, labels):
    cm = confusion_matrix(y, p, labels=labels)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), [l.replace(" and ", "\n& ").replace(" ", "\n", 1) if len(l) > 10 else l for l in labels], fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title(title, fontsize=10)
    fig.colorbar(im, fraction=0.046); fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)


plot_cm(test.category, pt["category"], "Category: confusion matrix (stratified split)", "reports/confusion_category_split.png", CATEGORIES)
plot_cm(hand.category, ph["category"], "Category: confusion matrix (hand-written set)", "reports/confusion_category_handwritten.png", CATEGORIES)
plot_cm(test.urgency, pt["urgency"], "Urgency: confusion matrix", "reports/confusion_urgency.png", ["High", "Medium", "Low"])

names = list(comparison)
x = np.arange(len(names)); w = 0.38
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.bar(x - w / 2, [comparison[n]["split"]["f1"] for n in names], w, label="Stratified split")
ax.bar(x + w / 2, [comparison[n]["handwritten"]["f1"] for n in names], w, label="Hand-written set")
ax.set_xticks(x, names, fontsize=8); ax.set_ylim(0, 1.05); ax.set_ylabel("Macro F1")
ax.set_title("Category classifier comparison"); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig("reports/model_comparison.png", dpi=160); plt.close(fig)

# ---------------------------------------------------------------- final model on ALL data -> deployment
final = GrievanceClassifier().fit(df.text, df.category, df.urgency, df.sentiment)
with open("models/model.pkl", "wb") as f:
    pickle.dump(final, f)

with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(json.dumps(metrics, indent=2))
