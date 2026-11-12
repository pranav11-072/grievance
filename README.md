# Kannada-English Citizen Grievance Classification and Response Framework

Course: 23CSI73 Natural Language Processing (CCE) | Domain: Integrated Domain Applications

A code-mixed (English / Romanized Kannada / Kannada script) complaint is classified by department, scored for
urgency and sentiment, mined for location and time, answered with a templated bilingual reply, and stored for an
officer dashboard. Low-confidence complaints are routed to a human instead of being auto-handled.

## Run it

```bash
pip install -r requirements.txt
python generate_dataset.py      # 872 synthetic labelled complaints -> data/complaints.csv
python train_evaluate.py        # trains, evaluates, writes models/model.pkl and reports/
python app.py                   # http://127.0.0.1:5000   (set PORT to change)
```

API: `POST /api/complaints {"text": "..."}`, `GET /api/complaints`, `GET /api/stats`.

## Deploy on Render

This repository includes a Render Blueprint in `render.yaml`.

1. Open [Render](https://render.com), choose **New → Blueprint**, and connect this repository.
2. Select the `main` branch and apply the Blueprint. Render will install `requirements.txt` and start the app with Gunicorn.
3. The service uses `/health` for health checks. The trained model is loaded from `models/model.pkl`; its scikit-learn runtime is pinned to **1.8.0** to match the saved model.
4. SQLite stores submissions under `/var/data/complaints.db`. The Blueprint attaches a 1 GB persistent disk at `/var/data`, so submitted complaints survive deploys and restarts.

**Important:** The Blueprint selects a paid Starter web service because Render persistent disks are available only on paid services. A disk-backed service is limited to one instance, which is why the Gunicorn configuration uses one worker. See [Render persistent disks](https://render.com/docs/disks).

The officer dashboard (`/dashboard`) and complaint APIs are currently unauthenticated. Treat this as a demonstration deployment only; add officer authentication and access controls before using it with real citizens or sensitive complaints.

## Pipeline and syllabus mapping

| Stage | File / function | Module |
|---|---|---|
| Unicode clean-up, PII masking, Kannada to Roman transliteration | `nlp_pipeline.clean_unicode`, `transliterate_kannada` | I, II |
| Sentence and word tokenization | `sent_tokenize`, `word_tokenize` | I |
| SMS expansion, spelling variants, edit-distance spell correction | `SMS_EXPAND`, `VARIANTS`, `SpellCorrector` | II |
| Stopwords, stemming (Snowball), lemmatization | `STOPWORDS`, `lemmatize` | I, II |
| POS tagging, noun-phrase chunking | `pos_tag`, `chunk` (NLTK `RegexpParser`) | II, III |
| NER: location (gazetteer + typo tolerance), landmark, time | `extract_location`, `extract_time` | III |
| Word-sense cues for "light", "cut", "pipe", "drain" | `disambiguate` | IV |
| TF-IDF (word 1-3 and char 2-5 grams) + Logistic Regression | `classifier.GrievanceClassifier` | I, V |
| Template-based reply generation (English and Kannada) | `response_generator.generate_reply` | V |

NLTK data packages could not be downloaded in the build sandbox, so the POS tagger uses a lexicon + regex
fallback automatically. On your own machine run `python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('wordnet')"`
and the pipeline switches to NLTK's perceptron tagger and WordNet lemmatizer with no code change.

## Results (read this before quoting numbers)

The training data is **synthetic**: complaints are generated from templates with noise. A random train/test split
therefore contains near-duplicates of training sentences and scores ~100%. That number shows the pipeline is wired
correctly, not that the system generalises. The honest number is the **hand-written test set** (36 complaints
written separately, sharing no templates with training).

| Category classifier (macro F1) | Stratified split (n=175) | Hand-written (n=36) |
|---|---|---|
| Rule-based keywords | 0.924 | 0.913 |
| TF-IDF + Naive Bayes | 1.000 | 0.884 |
| TF-IDF + Linear SVM | 1.000 | 0.910 |
| TF-IDF + Logistic Regression (deployed) | 1.000 | 0.885 |

Other headline figures: 5-fold CV macro F1 0.999 (synthetic, optimistic); sentiment accuracy 0.91 on the split but
only ~0.67 on hand-written text; **urgency accuracy 0.53 on hand-written text** (the model mostly memorised the
closing phrases used in the synthetic data). With the 0.55 confidence threshold, 3 of the 4 category errors on
the hand-written set were flagged for human review, and accuracy on the auto-handled remainder was 97%.

Observations worth putting in the report: the keyword baseline is competitive with ML on short, keyword-heavy
complaints, so the ML gain only shows up with a larger real dataset; Roads and Potholes is the weakest class
(its vocabulary overlaps with water and waste complaints, e.g. "pipe cut during road digging"); location extraction
is exact on the dataset because the gazetteer was built from the same place list, so it is optimistic too.

## Known limitations (be upfront about these in the viva)

- Replace the synthetic data with real or manually collected complaints before making any accuracy claim.
  Even 200 genuine complaints would be more informative than 872 synthetic ones.
- Urgency needs real labels; the current model is not reliable. The "emergency / accident / unsafe" keyword
  override is the safety net.
- Transliteration is dictionary-based (about 90 stems), not a full Kannada to Roman model.
- Place names outside the gazetteer are only found by the "X Nagar / X Layout / X Road" pattern.
- No authentication on the dashboard; add login before any real deployment. Complaint text can contain personal data.

## Ethics (for the report)

Human-in-the-loop routing for low confidence; no personal data is needed beyond the text; phone numbers and e-mails
are masked in the pipeline; the model predicts department and priority only and never decides whether a complaint
is valid; bilingual replies avoid excluding Kannada-only citizens.
