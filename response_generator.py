"""Template-based reply generation (Module V: NLG) in English and Kannada.

The reply language follows the complaint's language style: Kannada-script and Romanized Kannada
complaints get a bilingual reply (Kannada first), English complaints get English.
"""
import hashlib
import re
from datetime import datetime

DEPARTMENTS = {
    "Street Lighting": {"en": "Street Lighting Division", "kn": "ಬೀದಿ ದೀಪ ವಿಭಾಗಕ್ಕೆ", "sla": {"High": 24, "Medium": 72, "Low": 120}},
    "Water Supply": {"en": "Water Supply and Sewerage Board", "kn": "ನೀರು ಸರಬರಾಜು ಮಂಡಳಿಗೆ", "sla": {"High": 12, "Medium": 48, "Low": 96}},
    "Roads and Potholes": {"en": "Roads and Infrastructure Wing", "kn": "ರಸ್ತೆ ಮತ್ತು ಮೂಲಸೌಕರ್ಯ ವಿಭಾಗಕ್ಕೆ", "sla": {"High": 48, "Medium": 96, "Low": 168}},
    "Waste Collection": {"en": "Solid Waste Management Cell", "kn": "ಘನ ತ್ಯಾಜ್ಯ ನಿರ್ವಹಣಾ ವಿಭಾಗಕ್ಕೆ", "sla": {"High": 24, "Medium": 48, "Low": 96}},
}
ISSUE_EN = {"Street Lighting": "street light", "Water Supply": "water supply", "Roads and Potholes": "road damage",
            "Waste Collection": "garbage collection"}
ISSUE_KN = {"Street Lighting": "ಬೀದಿ ದೀಪ", "Water Supply": "ನೀರು ಸರಬರಾಜು", "Roads and Potholes": "ರಸ್ತೆ ಹಾನಿ",
            "Waste Collection": "ಕಸ ಸಂಗ್ರಹಣೆ"}
URGENCY_KN = {"High": "ತುರ್ತು", "Medium": "ಸಾಮಾನ್ಯ", "Low": "ಕಡಿಮೆ ಆದ್ಯತೆ"}


CODES = {"Street Lighting": "SL", "Water Supply": "WS", "Roads and Potholes": "RP", "Waste Collection": "WC"}


def make_ticket_id(text, category):
    h = hashlib.sha1(f"{text}{datetime.now().isoformat()}".encode()).hexdigest()[:5].upper()
    return f"GRV-{CODES[category]}-{h}"


def since_phrase(ent):
    """Turn the extracted time expression into a clean English phrase for the reply."""
    t = ent.get("time")
    if not t:
        return ""
    if re.match(r"(since|from|for)\b", t):
        return f" ({t})"
    m = re.search(r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", t)
    if m:
        return f" (since {m.group(1).title()})"
    if "ninne" in t or "yesterday" in t:
        return " (since yesterday)"
    n = re.search(r"(\d+|ondu|eradu)", t)
    if n:
        num = {"ondu": 1, "eradu": 2}.get(n.group(1), n.group(1))
        unit = "weeks" if "vaar" in t else "days"
        return f" (for {num} {unit})"
    if "vaar" in t:
        return " (for about a week)"
    return ""


def hours_text(h):
    return f"{h} hours" if h < 48 else f"{h // 24} working days"


def hours_text_kn(h):
    return f"{h} ಗಂಟೆಗಳೊಳಗೆ" if h < 48 else f"{h // 24} ಕೆಲಸದ ದಿನಗಳೊಳಗೆ"


def generate_reply(result, ticket_id):
    """result: dict from GrievanceClassifier.predict. Returns dict with en / kn / text."""
    cat, urg, sent = result["category"], result["urgency"], result["sentiment"]
    ent = result["analysis"]["entities"]
    style = result["analysis"]["language_style"]
    dept = DEPARTMENTS[cat]
    sla = dept["sla"][urg]
    loc_en = f" at {ent['location']}" if ent["location"] else ""
    loc_kn = f" {ent['location']} ಪ್ರದೇಶದಲ್ಲಿ" if ent["location"] else ""
    since_en = since_phrase(ent)

    apology_en = "We sincerely apologise for the inconvenience. " if sent == "Negative" else ""
    apology_kn = "ಅನಾನುಕೂಲಕ್ಕಾಗಿ ಕ್ಷಮೆ ಕೋರುತ್ತೇವೆ. " if sent == "Negative" else ""
    prio_en = {"High": "We have marked it as HIGH priority and alerted the field team. ",
               "Medium": "It has been assigned to the field team. ",
               "Low": "It has been added to the scheduled work list. "}[urg]
    prio_kn = {"High": "ಇದನ್ನು ತುರ್ತು ಎಂದು ಗುರುತಿಸಿ ಸಿಬ್ಬಂದಿಗೆ ತಿಳಿಸಲಾಗಿದೆ. ",
               "Medium": "ಇದನ್ನು ಕ್ಷೇತ್ರ ಸಿಬ್ಬಂದಿಗೆ ನಿಯೋಜಿಸಲಾಗಿದೆ. ",
               "Low": "ಇದನ್ನು ನಿಗದಿತ ಕೆಲಸದ ಪಟ್ಟಿಗೆ ಸೇರಿಸಲಾಗಿದೆ. "}[urg]

    en = (f"Dear Citizen, thank you for reporting the {ISSUE_EN[cat]} issue{loc_en}{since_en}. "
          f"{apology_en}{prio_en}Your complaint (Ticket {ticket_id}) has been forwarded to the {dept['en']}, "
          f"and we expect to resolve it within {hours_text(sla)}.")
    if result.get("needs_review"):
        en += " A grievance officer will also review your complaint personally to confirm the right department."
    kn = (f"ಆತ್ಮೀಯ ನಾಗರಿಕರೇ, {loc_kn} {ISSUE_KN[cat]} ಸಮಸ್ಯೆಯನ್ನು ವರದಿ ಮಾಡಿದ್ದಕ್ಕಾಗಿ ಧನ್ಯವಾದಗಳು. "
          f"{apology_kn}{prio_kn}ನಿಮ್ಮ ದೂರು (ಟಿಕೆಟ್ {ticket_id}) ಅನ್ನು {dept['kn']} ಕಳುಹಿಸಲಾಗಿದೆ; "
          f"{hours_text_kn(sla)} ಪರಿಹರಿಸಲು ಪ್ರಯತ್ನಿಸುತ್ತೇವೆ. ಆದ್ಯತೆ: {URGENCY_KN[urg]}.").replace("  ", " ")
    text = en if style == "english" else f"{kn}\n\n{en}"
    return {"en": en, "kn": kn, "text": text, "department": dept["en"], "sla_hours": sla}
