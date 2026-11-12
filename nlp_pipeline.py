"""NLP pipeline for code-mixed Kannada-English citizen grievances.

Stages (syllabus mapping in brackets):
  1. Unicode clean-up + Kannada-script -> Romanized normalization   [II, transliteration]
  2. Word / sentence tokenization (regex tokenizer, no corpus needed)  [I, Lab 1]
  3. SMS-shorthand expansion + edit-distance spell correction          [II, Labs 3 and 5]
  4. Stopword removal (bundled English + Kanglish list)                [Lab 2]
  5. Stemming (Snowball) and lemmatization (WordNet if available,
     rule-based fallback)                                              [II]
  6. POS tagging (NLTK perceptron if installed, else lexicon+regex)    [II, Lab 4]
  7. Chunking (NLTK RegexpParser) and NER: location, landmark, time    [III, Labs 6 and 7]
  8. Ambiguity cues for "light", "cut", "pipe", "drain" (WSD, rules)   [IV]
"""
import re
import unicodedata

from nltk import RegexpParser, RegexpTagger
from nltk.metrics import edit_distance
from nltk.stem import SnowballStemmer

# ----------------------------------------------------------------------------------------
# Resources
# ----------------------------------------------------------------------------------------
LOCATIONS = [
    "MG Road", "Indiranagar 1st Stage", "Jayanagar 4th Block", "Koramangala", "BTM Layout",
    "Whitefield", "HSR Layout", "Malleshwaram", "Rajajinagar", "Yelahanka", "Banashankari",
    "Basavanagudi", "Hebbal", "Marathahalli", "JP Nagar", "Electronic City", "Vijayanagar",
    "KR Puram", "Shivajinagar", "Bannerghatta Road", "Kengeri", "Sadashivanagar", "Peenya",
    "Domlur", "Ulsoor", "Mysore Road", "Hennur", "Kalyan Nagar",
]
LANDMARKS = ["bus stop", "temple", "school", "park", "market", "hospital", "metro station"]
LOC_SUFFIX = {"road", "nagar", "layout", "block", "stage", "circle", "cross", "colony", "extension",
              "main", "street", "city", "pura", "halli"}

EN_STOP = set("""a an the and or but if of at by for with about against between into through during before
after above below to from up down in out on off over under again further then once here there when where
why how all any both each few more most other some such no nor not only own same so than too very s t can
will just don should now is are was were be been being have has had having do does did doing i me my we our
you your he she it its they them their this that these those am would could may might shall also""".split())
# Romanized Kannada function words / fillers (kept out of the feature space)
KN_STOP = set("""alli nalli nali ge ge ella ellaa kade hattira hatira inda dayavittu dayavitu dayavittu please plz
pls yenu ide ive aagide aagutte aagtide mattu mathu ondu namma nimma ivattu idu adu ee aa""".split())
STOPWORDS = EN_STOP | KN_STOP
# Keep negations and intensifiers: they matter for urgency / sentiment
KEEP = {"not", "no", "nor", "very", "too", "off"}
STOPWORDS -= KEEP

SMS_EXPAND = {
    "plz": "please", "pls": "please", "stret": "street", "lite": "light", "lites": "lights",
    "watr": "water", "garbge": "garbage", "pothol": "pothole", "rd": "road", "nt": "not",
    "sinc": "since", "urgnt": "urgent", "urgntly": "urgently", "suply": "supply", "colected": "collected",
    "workin": "working", "u": "you", "ur": "your", "wid": "with", "d": "the", "yday": "yesterday",
    "tmrw": "tomorrow", "asap": "urgent", "msg": "message", "govt": "government",
}

# Dictionary-based transliteration of Kannada-script stems into the Romanized forms used in text.
KANNADA_MAP = {
    "ನೀರು": "neeru", "ದೀಪ": "deepa", "ಬೀದಿ": "street", "ಗುಂಡಿ": "gundi", "ರಸ್ತೆ": "raste", "ಕಸದ": "kasa",
    "ಕಸ": "kasa", "ಕತ್ತಲೆ": "kattale", "ಉರಿಯುತ್ತಿಲ್ಲ": "uriyuttilla", "ಪೈಪ್": "pipe", "ಒಡೆ": "odedu",
    "ಸೋರು": "soru", "ಕುಡಿಯುವ": "kudiyuva", "ಕೊಳಕಾಗಿದೆ": "kolachaagide", "ಕೊಳಕು": "kolachaagide",
    "ವಾಸನೆ": "vaasane", "ತೆಗೆಯುತ್ತಿಲ್ಲ": "tegiyalla", "ಬರುತ್ತಿಲ್ಲ": "barthilla", "ಹಾಳಾಗಿ": "haalaagide",
    "ಗಾಡಿ": "vaahana", "ರಾಶಿ": "raasi", "ಮ್ಯಾನ್‌ಹೋಲ್": "manhole", "ಮ್ಯಾನ್ಹೋಲ್": "manhole",
    "ಮುಚ್ಚಳ": "mucchala", "ತಕ್ಷಣ": "thakshana", "ತುರ್ತು": "urgent", "ಅಪಘಾತ": "accident",
    "ಮಕ್ಕಳಿಗೆ": "makkalige", "ಬೇಗ": "bega", "ಹತ್ತಿರ": "hattira", "ನಲ್ಲಿ": "nalli", "ಕೆಲಸ": "work",
    "ಮಾಡುತ್ತಿಲ್ಲ": "not-working", "ಸರಿಪಡಿಸಿ": "sari-maadi", "ಕ್ರಮ": "action", "ಕೈಗೊಳ್ಳಿ": "togoli",
    "ದಯವಿಟ್ಟು": "dayavittu", "ತೊಂದರೆ": "tondare", "ತುಂಬಾ": "tumba", "ಕಷ್ಟವಾಗಿದೆ": "kasta-aagide",
    "ಸಹಿಸಲು": "sahisoke", "ಧನ್ಯವಾದ": "dhanyavaada", "ಕಾಳಜಿ": "kaalji", "ಇಲ್ಲ": "illa", "ಇದೆ": "ide",
    "ಬಿದ್ದಿದೆ": "biddide", "ನಮ್ಮ": "namma", "ದೊಡ್ಡ": "dodda", "ಏರಿಯಾ": "area", "ಪ್ರದೇಶದಲ್ಲಿ": "area",
    "ಎಲ್ಲೆಡೆ": "ella-kade", "ಹಾಳಾಗಿವೆ": "haalaagide", "ದೀಪಗಳು": "deepa", "ಗುಂಡಿಗಳು": "gundi",
    "ಸೋಮವಾರ": "monday", "ಮಂಗಳವಾರ": "tuesday", "ಭಾನುವಾರ": "sunday", "ನಿನ್ನೆ": "yesterday",
    "ಕಳೆದ": "last", "ವಾರ": "week", "ದಿನ": "day", "ಮೂರು": "3", "ಐದು": "5", "ಎರಡು": "2",
    # place names (Kannada script -> English gazetteer spelling)
    "ಜಯನಗರ": "Jayanagar", "ಕೋರಮಂಗಲ": "Koramangala", "ಹೆಬ್ಬಾಳ": "Hebbal", "ಇಂದಿರಾನಗರ": "Indiranagar",
    "ವೈಟ್\u200cಫೀಲ್ಡ್": "Whitefield", "ಮಲ್ಲೇಶ್ವರ": "Malleshwaram", "ರಾಜಾಜಿನಗರ": "Rajajinagar",
    "ಯಲಹಂಕ": "Yelahanka", "ಬನಶಂಕರಿ": "Banashankari", "ಬಸವನಗುಡಿ": "Basavanagudi", "ಮಾರತ್ತಹಳ್ಳಿ": "Marathahalli",
    "ವಿಜಯನಗರ": "Vijayanagar", "ಕೆಂಗೇರಿ": "Kengeri", "ಪೀಣ್ಯ": "Peenya", "ಡೊಮ್ಲೂರು": "Domlur",
}
KANNADA_KEYS = sorted(KANNADA_MAP, key=len, reverse=True)

# Romanized Kannada spelling variants -> one canonical spelling (rule-based normalization)
VARIANTS = {
    "baralla": "barthilla", "barlilla": "barthilla", "baralilla": "barthilla", "bartilla": "barthilla",
    "barthilla": "barthilla", "baratilla": "barthilla",
    "uriyalla": "uriyuttilla", "uriyutilla": "uriyuttilla", "uriyuttilla": "uriyuttilla",
    "gundi": "gundi", "gundigalu": "gundi", "gundige": "gundi",
    "neru": "neeru", "niru": "neeru", "neer": "neeru",
    "kasa": "kasa", "kasada": "kasa", "kasavu": "kasa",
    "raste": "raste", "rasthe": "raste", "rasta": "raste",
    "kattale": "kattale", "katale": "kattale", "kathale": "kattale",
    "deepa": "deepa", "deepada": "deepa", "dipa": "deepa", "kambada": "kambada",
    "tegiyalla": "tegiyalla", "tegiyiri": "tegiyiri", "tegiyodilla": "tegiyalla",
    "vaasane": "vaasane", "vasane": "vaasane", "vaasne": "vaasane",
    "tondare": "tondare", "tondre": "tondare", "thondare": "tondare",
    "odediddhe": "odedide", "odedide": "odedide", "odediddare": "odedide",
}

DAYS = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"}
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "a": 1, "ondu": 1,
             "eradu": 2, "mooru": 3, "naalku": 4, "aidu": 5}

# Lexicons used for urgency cues and sentiment features
URGENT_CUES = {"urgent", "urgently", "emergency", "immediately", "immediate", "asap", "unsafe", "accident",
               "dangerous", "dangerously", "thakshana", "apaayakaari", "aksidenta", "tondare", "children",
               "elderly", "makkalige", "bega", "risk", "suffering", "pitch", "dark"}
NEG_CUES = {"worst", "frustrating", "unacceptable", "fed", "negligence", "shame", "disappointed", "kasta",
            "bejaar", "sahisoke", "kopa", "terrible", "dirty", "stink", "smell", "vaasane", "kaalji",
            "not", "no", "dead", "broken", "burst"}
POS_CUES = {"thank", "thanks", "thankful", "grateful", "dhanyavadagalu", "dhanyavaada", "regards", "kindly"}

# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[ಀ-೿‌‍]+|\d+(?:st|nd|rd|th)?|[A-Za-z]+(?:'[a-z]+)?|[^\w\s]")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def sent_tokenize(text):
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def word_tokenize(text):
    return _TOKEN_RE.findall(text)


def clean_unicode(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"\b\d{10}\b", " <phone> ", text)             # anonymise phone numbers
    text = re.sub(r"[\w.]+@[\w.]+", " <email> ", text)          # anonymise e-mails
    text = re.sub(r"(.)\1{3,}", r"\1\1", text)                  # sooooo -> soo
    return re.sub(r"\s+", " ", text).strip()


def transliterate_kannada(text):
    """Map Kannada-script tokens to Romanized stems (dictionary based, longest prefix first)."""
    out = []
    for tok in re.findall(r"[ಀ-೿‌‍]+|[^ಀ-೿‌‍]+", text):
        if not re.match(r"[ಀ-೿]", tok):
            out.append(tok)
            continue
        for key in KANNADA_KEYS:
            if tok.startswith(key):
                rest = tok[len(key):]
                piece = KANNADA_MAP[key]
                if "ಂದ" in rest:                              # ablative "-inda" (since / from)
                    piece += " inda"
                out.append(piece)
                break
        else:
            out.append(tok)                                    # unknown Kannada word: keep as is
    return re.sub(r"\s+", " ", "".join(out)).strip()


def _split_inflect(word):
    return re.sub(r"(ies)$", "y", word) if len(word) > 4 else word


# ----------------------------------------------------------------------------------------
# Spell corrector (Lab 5): bounded edit distance against a corpus vocabulary
# ----------------------------------------------------------------------------------------
class SpellCorrector:
    def __init__(self, vocab=None):
        self.vocab = set(vocab or [])
        self.protected = {w.lower() for loc in LOCATIONS for w in loc.split()} | set(LANDMARKS)

    def fit(self, token_lists, min_count=4):
        from collections import Counter
        c = Counter(t for toks in token_lists for t in toks if t.isalpha())
        self.vocab = {w for w, n in c.items() if n >= min_count}
        return self

    def correct(self, tok):
        if (tok in self.vocab or not tok.isalpha() or len(tok) < 5 or tok in self.protected
                or not self.vocab):
            return tok
        limit = 1 if len(tok) < 8 else 2
        best, best_d = tok, limit + 1
        for cand in self.vocab:
            if abs(len(cand) - len(tok)) > limit or cand[0] != tok[0]:
                continue
            d = edit_distance(tok, cand, transpositions=True)
            if d < best_d or (d == best_d and cand < best):
                best, best_d = cand, d
        return best if best_d <= limit else tok


# ----------------------------------------------------------------------------------------
# POS tagger: NLTK perceptron if its data is installed, otherwise a lexicon + regex tagger
# ----------------------------------------------------------------------------------------
_LEXICON = {
    **{w: "DT" for w in ["the", "a", "an", "this", "that", "these", "those", "our", "my", "your", "all", "every"]},
    **{w: "IN" for w in ["near", "in", "at", "on", "from", "since", "for", "of", "by", "with", "without",
                         "outside", "beside", "inside", "behind", "after", "before", "during"]},
    **{w: "CC" for w in ["and", "or", "but"]},
    **{w: "PRP" for w in ["it", "we", "they", "i", "you", "he", "she", "us"]},
    **{w: "TO" for w in ["to"]},
    **{w: "MD" for w in ["can", "will", "could", "should", "would", "may", "must"]},
    **{w: "VBZ" for w in ["is", "has", "does", "keeps", "goes", "comes"]},
    **{w: "VBP" for w in ["are", "have", "do"]},
    **{w: "VBD" for w in ["was", "did", "came", "went"]},
    **{w: "VBN" for w in ["been", "worked", "fixed", "dumped", "burst", "broken", "damaged", "collected",
                          "supplied", "left", "filled", "flooded", "fused", "cut", "restored", "blocked"]},
    **{w: "VB" for w in ["fix", "repair", "clear", "clean", "replace", "check", "resolve", "look", "see", "buy",
                         "work", "come", "take", "act", "remove", "restore"]},
    **{w: "RB" for w in ["not", "urgently", "immediately", "soon", "now", "very", "again", "too", "only", "dangerously",
                         "completely", "badly", "yesterday", "today", "always"]},
    **{w: "JJ" for w in ["urgent", "dark", "dirty", "smelly", "huge", "deep", "broken", "dead", "unsafe", "low",
                         "terrible", "dry", "contaminated", "open", "main", "unacceptable", "full", "loose"]},
    **{w: "NNP" for w in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]},
    "please": "UH",
}
_REGEX_RULES = [
    (r"^\d+(st|nd|rd|th)?$", "CD"), (r"^(one|two|three|four|five|six|seven)$", "CD"),
    (r".*ing$", "VBG"), (r".*ed$", "VBD"), (r".*ly$", "RB"), (r".*(ous|ful|ive|able|al)$", "JJ"),
    (r".*s$", "NNS"), (r"^[.!?]$", "."), (r"^[,;:]$", ","), (r".*", "NN"),
]
_FALLBACK_TAGGER = RegexpTagger(_REGEX_RULES)


def pos_tag(tokens):
    try:
        from nltk import pos_tag as nltk_pos_tag
        return nltk_pos_tag(tokens)
    except Exception:                                           # LookupError: data not installed
        tagged = []
        for i, tok in enumerate(tokens):
            low = tok.lower()
            if low in _LEXICON:
                tagged.append((tok, _LEXICON[low]))
            elif tok[:1].isupper() and i > 0 and tok.isalpha():
                tagged.append((tok, "NNP"))
            else:
                tagged.append((tok, _FALLBACK_TAGGER.tag([low])[0][1]))
        return tagged


_CHUNKER = RegexpParser(r"""
    NP: {<DT|PRP\$>?<JJ.*>*<NN.*>+}
    PP: {<IN><NP>}
""")


def chunk(tagged):
    tree = _CHUNKER.parse(tagged)
    return [" ".join(w for w, _ in st.leaves()) for st in tree.subtrees() if st.label() == "NP"]


# ----------------------------------------------------------------------------------------
# Lemmatization
# ----------------------------------------------------------------------------------------
_IRREGULAR = {"worked": "work", "working": "work", "lights": "light", "potholes": "pothole", "dumped": "dump",
              "flooded": "flood", "collected": "collect", "supplied": "supply", "damaged": "damage",
              "leaking": "leak", "piled": "pile", "piling": "pile", "burning": "burn", "burst": "burst",
              "came": "come", "coming": "come", "broken": "break", "urgently": "urgent", "tegiyalla": "tege",
              "pipes": "pipe", "roads": "road", "streets": "street", "days": "day", "weeks": "week"}
_stemmer = SnowballStemmer("english")


def lemmatize(tok):
    if tok in _IRREGULAR:
        return _IRREGULAR[tok]
    try:
        from nltk.stem import WordNetLemmatizer
        lem = WordNetLemmatizer()
        v = lem.lemmatize(tok, "v")
        return v if v != tok else lem.lemmatize(tok, "n")
    except Exception:
        for suf, rep in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
            if tok.endswith(suf) and len(tok) - len(suf) >= 3:
                return tok[: len(tok) - len(suf)] + rep
        return tok


# ----------------------------------------------------------------------------------------
# Entity extraction
# ----------------------------------------------------------------------------------------
_LOC_TOKENS = [(loc, [w.lower() for w in loc.split()]) for loc in LOCATIONS]
_LOC_TOKENS.sort(key=lambda x: -len(x[1]))
BASE_AREAS = {w.lower() for w in ["Jayanagar", "Indiranagar", "Koramangala", "Whitefield", "Malleshwaram",
              "Rajajinagar", "Yelahanka", "Banashankari", "Basavanagudi", "Hebbal", "Marathahalli", "Vijayanagar",
              "Shivajinagar", "Kengeri", "Sadashivanagar", "Peenya", "Domlur", "Ulsoor", "Hennur"]}


def extract_location(tokens_lower):
    """Gazetteer match (longest first, tolerant to one typo), then pattern fall-back."""
    n = len(tokens_lower)
    for loc, parts in _LOC_TOKENS:
        k = len(parts)
        for i in range(n - k + 1):
            window = tokens_lower[i:i + k]
            if window == parts:
                return loc
            if k == 1 and len(parts[0]) >= 6 and window[0].isalpha() \
                    and edit_distance(window[0], parts[0], transpositions=True) <= 1:
                return loc
            if k > 1 and " ".join(window) and edit_distance(" ".join(window), " ".join(parts),
                                                           transpositions=True) <= 1:
                return loc
    for tok in tokens_lower:                       # bare area name, e.g. "jayanagar nalli ..."
        if tok in BASE_AREAS:
            return tok.title()
    for i, tok in enumerate(tokens_lower):
        if tok in LOC_SUFFIX and i > 0 and tokens_lower[i - 1].isalpha() \
                and tokens_lower[i - 1] not in STOPWORDS | {"main", "the", "bus", "road", "this", "that"}:
            return f"{tokens_lower[i - 1].title()} {tok.title()}"
    return None


def extract_landmark(tokens_lower):
    text = " ".join(tokens_lower)
    for lm in LANDMARKS:
        if re.search(rf"\b{re.escape(lm)}\b", text):
            return lm
    return None


_TIME_PATTERNS = [
    r"(?:since|from)\s+(?:last\s+)?(?:%s|yesterday|week|monday|today)" % "|".join(DAYS),
    r"(?:for|from)\s+(?:the\s+)?(?:last\s+)?(?:\d+|one|two|three|four|five|six|seven|a)\s+(?:days?|weeks?|months?)(?:\s+now)?",
    r"(?:%s|ninne|yesterday|last week)\s+inda" % "|".join(DAYS),
    r"(?:kaledha\s+)?(?:\d+|ondu|eradu|mooru|3|5|2)\s+(?:dinagalinda|vaaradinda|vaaragalinda|days?|day|week)\s*(?:inda)?",
    r"(?:ondu|eradu)\s+vaarada?\s+inda",
    r"kaledha\s+vaaradinda",
    r"(?:last|since)\s+(?:night|week|month)",
    r"yesterday", r"today", r"tomorrow",
]
_TIME_RE = re.compile("|".join(f"(?:{p})" for p in _TIME_PATTERNS))


def extract_time(norm_text):
    m = _TIME_RE.search(norm_text)
    return m.group(0).strip() if m else None


def duration_days(time_str):
    """Rough number of days an issue has been going on, used as an urgency signal."""
    if not time_str:
        return 0
    m = re.search(r"(\d+|one|two|three|four|five|six|seven|ondu|eradu|a)\s+(day|week|vaara|dina)", time_str)
    if m:
        raw = m.group(1)
        n = int(raw) if raw.isdigit() else NUM_WORDS.get(raw, 1)
        return n * (7 if m.group(2) in ("week", "vaara") else 1)
    if "week" in time_str or "vaaradinda" in time_str:
        return 7
    if any(d in time_str for d in DAYS):
        return 3
    if "yesterday" in time_str or "ninne" in time_str:
        return 1
    return 0


# ----------------------------------------------------------------------------------------
# Word-sense cues (Module IV): resolve "light", "cut", "pipe", "drain" using neighbours
# ----------------------------------------------------------------------------------------
def disambiguate(tokens):
    senses = {}
    s = set(tokens)
    if "light" in s or "lights" in s or "lamp" in s:
        if s & {"street", "pole", "lamp", "dark", "bulb", "kattale", "deepa", "traffic", "road"}:
            senses["light"] = "street light (public lighting)"
        elif s & {"rain", "weight", "load"}:
            senses["light"] = "not heavy (not a lighting issue)"
        else:
            senses["light"] = "street light (public lighting)"
    if "cut" in s:
        if s & {"water", "neeru", "tap", "supply"}:
            senses["cut"] = "supply interruption (water)"
        elif s & {"road", "cable", "pipe", "raste"}:
            senses["cut"] = "excavation / cut (roads)"
        elif s & {"tree", "branches"}:
            senses["cut"] = "tree cutting (waste)"
        elif s & {"light", "power", "current"}:
            senses["cut"] = "power outage (electrical)"
    if "pipe" in s and s & {"road", "cable", "contractors", "raste"} and "cut" in s:
        senses["pipe"] = "water pipeline damaged by digging"
    if "drain" in s:
        senses["drain"] = "drainage (road/waste)" if s & {"garbage", "waste", "kasa", "dump", "trash"} \
            else "drain cover / road hazard"
    return senses


# ----------------------------------------------------------------------------------------
# The pipeline
# ----------------------------------------------------------------------------------------
class Pipeline:
    def __init__(self, spell=None):
        self.spell = spell or SpellCorrector()

    # --- normalization used for model features (string in, string out) -------------------
    def normalise_tokens(self, text, correct=True):
        text = clean_unicode(text)
        text = transliterate_kannada(text)
        toks = [t.lower() for t in word_tokenize(text)]
        toks = [SMS_EXPAND.get(t, t) for t in toks]
        toks = [VARIANTS.get(t, t) for t in toks]
        if correct:
            toks = [VARIANTS.get(self.spell.correct(t), t) for t in toks]
        return toks

    def feature_text(self, text):
        """Cleaned, stopword-free, stemmed string used for TF-IDF."""
        toks = self.normalise_tokens(text)
        loc = extract_location(toks)
        if loc:  # mask the place name so "MG Road" / "Mysore Road" cannot leak into the category
            masked = {w.lower() for w in loc.split()}
            toks = [t for t in toks if t not in masked]
        toks = [t for t in toks if t.isalnum() and t not in STOPWORDS]
        return " ".join(_stemmer.stem(t) if t.isascii() else t for t in toks)

    # --- full analysis for the demo / API ----------------------------------------------
    def analyse(self, text):
        cleaned = clean_unicode(text)
        roman = transliterate_kannada(cleaned)
        sentences = sent_tokenize(roman)
        raw_tokens = word_tokenize(roman)
        norm = self.normalise_tokens(text)
        content = [t for t in norm if t.isalnum() and t not in STOPWORDS]
        lemmas = [lemmatize(t) for t in content]
        stems = [_stemmer.stem(t) for t in content if t.isascii()]
        tagged = pos_tag([t for t in norm if t.isalnum()])
        norm_text = " ".join(norm)
        location = extract_location(norm)
        landmark = extract_landmark(norm)
        time = extract_time(norm_text)
        return {
            "original": text,
            "cleaned": cleaned,
            "romanized": roman,
            "sentences": sentences,
            "tokens": raw_tokens,
            "normalised_tokens": norm,
            "content_tokens": content,
            "lemmas": lemmas,
            "stems": stems,
            "pos_tags": tagged,
            "noun_chunks": chunk(tagged),
            "entities": {"location": location, "landmark": landmark, "time": time},
            "duration_days": duration_days(time),
            "senses": disambiguate(norm),
            "urgent_cues": sorted(set(norm) & URGENT_CUES),
            "negative_cues": sorted(set(norm) & NEG_CUES),
            "polite_cues": sorted(set(norm) & POS_CUES),
            "language_style": detect_language_style(text, norm),
        }


def detect_language_style(original, norm_tokens):
    if re.search(r"[ಀ-೿]", original):
        return "kannada_script"
    kn_markers = {"alli", "nalli", "nali", "hattira", "barthilla", "uriyuttilla", "tegiyalla", "inda", "neeru",
                  "kasa", "raste", "gundi", "sari", "maadi", "tondare", "namma", "ide", "aagide", "dayavittu",
                  "bega", "tumba", "dhanyavadagalu", "kattale", "deepa", "kambada", "ge"}
    hits = len(set(norm_tokens) & kn_markers)
    return "kanglish" if hits >= 2 else "english"
