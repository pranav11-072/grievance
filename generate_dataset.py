"""Generate a synthetic labelled dataset of citizen grievances.

Styles: English, Romanized Kannada code-mixed ("kanglish"), Kannada script code-mixed.
Labels: category, urgency, sentiment. Also stores gold location/time for NER evaluation.
Noise (typos, SMS shorthand, dropped punctuation) is injected to mimic real complaints.

Run:  python generate_dataset.py
"""
import csv
import random

random.seed(42)

CATEGORIES = ["Street Lighting", "Water Supply", "Roads and Potholes", "Waste Collection"]

LOCATIONS = [
    "MG Road", "Indiranagar 1st Stage", "Jayanagar 4th Block", "Koramangala", "BTM Layout",
    "Whitefield", "HSR Layout", "Malleshwaram", "Rajajinagar", "Yelahanka", "Banashankari",
    "Basavanagudi", "Hebbal", "Marathahalli", "JP Nagar", "Electronic City", "Vijayanagar",
    "KR Puram", "Shivajinagar", "Bannerghatta Road", "Kengeri", "Sadashivanagar", "Peenya",
    "Domlur", "Ulsoor", "Mysore Road", "Hennur", "Kalyan Nagar",
]
LANDMARKS = ["bus stop", "temple", "school", "park", "market", "hospital", "metro station"]

TIMES_EN = ["since Monday", "since yesterday", "for the last 3 days", "for 5 days", "since last week",
            "since Tuesday", "for two weeks", "since Sunday", "from Friday", "for a week now"]
TIMES_KN = ["Monday inda", "ninne inda", "kaledha 3 dinagalinda", "5 dinagalinda", "kaledha vaaradinda",
            "Tuesday inda", "eradu vaaragalinda", "Sunday inda", "Friday inda", "ondu vaarada inda"]
TIMES_KA = ["ಸೋಮವಾರದಿಂದ", "ನಿನ್ನೆಯಿಂದ", "ಮೂರು ದಿನಗಳಿಂದ", "ಐದು ದಿನಗಳಿಂದ", "ಕಳೆದ ವಾರದಿಂದ",
            "ಮಂಗಳವಾರದಿಂದ", "ಎರಡು ವಾರಗಳಿಂದ", "ಭಾನುವಾರದಿಂದ"]

# ---- urgency phrases -------------------------------------------------------------------
URG_EN = {
    "High": ["Please fix it urgently.", "This is an emergency!", "Immediate action needed.",
             "Accident can happen any time.", "Children and elderly people are suffering.",
             "It is unsafe, resolve ASAP.", "Very urgent, kindly act today."],
    "Medium": ["Kindly look into it soon.", "Please resolve this at the earliest.",
               "Request you to take action.", "Please do the needful."],
    "Low": ["Please check when possible.", "Just informing the department.",
            "Kindly look into this when you get time.", "No rush, but please see to it."],
}
URG_KN = {
    "High": ["thakshana sari maadi.", "tumba urgent ide, bega action togoli.",
             "makkalige tondare aagutte, urgent.", "aksidenta aagbahudu, dayavittu bega maadi.",
             "emergency, ivattu ne sari maadi."],
    "Medium": ["dayavittu bega sari maadi.", "dayavittu gamana kodi.", "action togolli please."],
    "Low": ["samaya sikkaaga nodi.", "gamanakke tandidini, nodi.", "tumba hurry illa, nodi."],
}
URG_KA = {
    "High": ["ತಕ್ಷಣ ಸರಿಪಡಿಸಿ.", "ತುಂಬಾ ತುರ್ತು, ಅಪಘಾತ ಆಗಬಹುದು.", "ಮಕ್ಕಳಿಗೆ ತೊಂದರೆ, ಬೇಗ ಕ್ರಮ ಕೈಗೊಳ್ಳಿ."],
    "Medium": ["ದಯವಿಟ್ಟು ಬೇಗ ಸರಿಪಡಿಸಿ.", "ಕ್ರಮ ಕೈಗೊಳ್ಳಿ."],
    "Low": ["ಸಮಯ ಸಿಕ್ಕಾಗ ನೋಡಿ.", "ಗಮನಕ್ಕೆ ತರುತ್ತಿದ್ದೇನೆ."],
}

# ---- complaint bodies ------------------------------------------------------------------
# slots: {loc} {lm} {t}
EN = {
    "Street Lighting": [
        "The street light near {loc} has not worked {t}.",
        "Street lights are not working on the road near {lm} in {loc}.",
        "The lamp post outside my house in {loc} is dead {t}, whole street is dark.",
        "Streetlight pole near {lm}, {loc} keeps flickering and then goes off.",
        "No street lights in our lane at {loc} {t}. It is pitch dark at night.",
        "Broken street lamp hanging dangerously near {lm} in {loc}.",
        "Street light is on during the day and wasting power in {loc}.",
        "The light on the main road at {loc} is fused {t}.",
        "Dark stretch near {lm}, {loc}. All the lights are off, women feel unsafe.",
        "Please replace the bulb of the street light at {loc}, it is off {t}.",
    ],
    "Water Supply": [
        "There is no water supply in {loc} {t}.",
        "Water is not coming in the taps at {loc} {t}.",
        "Drinking water is dirty and smelly in {loc}, tap water is brownish.",
        "Main water pipe is leaking near {lm} in {loc}, water is being wasted.",
        "Low water pressure in our apartment at {loc} {t}.",
        "Water cut in {loc} without any notice {t}.",
        "Pipeline burst near {lm}, {loc}. Road is flooded with water.",
        "Cauvery water has not been supplied to {loc} {t}.",
        "We are forced to buy tanker water as the tap is dry {t} in {loc}.",
        "Contaminated water coming from the borewell tap at {loc}.",
    ],
    "Roads and Potholes": [
        "There is a huge pothole on the road near {loc}.",
        "The road at {loc} is badly damaged and full of potholes {t}.",
        "Deep pothole near {lm} in {loc}, two-wheelers are slipping.",
        "Road repair work left incomplete near {lm}, {loc}, loose gravel everywhere.",
        "Potholes filled with rainwater on the main road at {loc}.",
        "Road cut for cable laying was not restored at {loc} {t}.",
        "Broken footpath and uneven road near {lm} in {loc}.",
        "The tar road in {loc} has completely come off, please re-asphalt.",
        "Speed breaker near {lm} at {loc} is broken and dangerous.",
        "Manhole cover missing on the road at {loc}, open pit.",
    ],
    "Waste Collection": [
        "Garbage has not been collected in {loc} {t}.",
        "The garbage truck did not come to {loc} {t}, waste is piling up.",
        "Overflowing dustbin near {lm} in {loc}, terrible smell.",
        "Waste is dumped on the roadside near {lm}, {loc}, stray dogs everywhere.",
        "Garbage is burning in the open at {loc}, lots of smoke.",
        "No door to door waste collection in our street at {loc} {t}.",
        "Dead animal and rotting garbage near {lm} in {loc}.",
        "Please clear the garbage heap at {loc}, it is spreading mosquitoes.",
        "Dry waste and wet waste are being mixed by the collectors at {loc}.",
        "Trash piled up beside the drain near {lm}, {loc}.",
    ],
}
KN = {
    "Street Lighting": [
        "{loc} hattira street light {t} uriyuttilla.",
        "{loc} alli {lm} hattira kambada deepa kettide, ella kade kattale.",
        "Namma street nalli light barthilla {t}, {loc}.",
        "{loc} nalli street lamp aarsi aarsi uriyutte, sari maadi.",
        "{lm} hattira dari deepa uriyalla {t}, {loc}.",
        "Raatri {loc} nalli tumba kattale, street light kharaabu aagide.",
        "{loc} main road nalli light fuse aagide {t}.",
        "Kambada deepa muri hogide {loc} alli, tondare ide.",
        "Hagalu kooda street light uriyuttide {loc} nalli, current waste aagtide.",
        "{loc} {lm} hattira lights elladu off, hengasarige suraksha illa.",
    ],
    "Water Supply": [
        "{loc} nalli neeru barthilla {t}.",
        "Namma area {loc} nalli {t} neeru baralla, tumba tondare.",
        "{loc} alli kudiyuva neeru kolachaagide, vaasane barutte.",
        "{lm} hattira main pipe odediddhe, neeru sorutide, {loc}.",
        "Tap nalli neeru pressure tumba kadime ide {t}, {loc}.",
        "{loc} nalli yaavude suchane illade water cut maadidru {t}.",
        "{loc} ge Kaveri neeru {t} barlilla, tanker beku.",
        "Pipeline odedu raste ella neeru, {lm} hattira {loc}.",
        "Borewell neeru kalusha aagide {loc} nalli, kudiyoke aagalla.",
        "Neeru supply illa {t}, tanker ge duddu kodtidini, {loc}.",
    ],
    "Roads and Potholes": [
        "{loc} hattira raste mele dodda gundi ide.",
        "{loc} nalli raste tumba haalaagide, gundigalu ive {t}.",
        "{lm} hattira aala gundi ide, bike gale jaarutive, {loc}.",
        "Raste kelasa ardhakke bittidare {lm} hattira {loc} nalli, jalli kallu ellaa kade.",
        "Male neeru gundi nalli tumbide, {loc} main road.",
        "Cable haakalu raste agediddare, {t} sari maadilla, {loc}.",
        "{loc} nalli footpath odedide, raste asamaana ide.",
        "{loc} nalli tar raste poorti hogide, hosa asphalt haaki.",
        "{lm} hattira speed breaker odedide, apaayakaari, {loc}.",
        "{loc} nalli manhole mucchala illa, raste mele gundi.",
    ],
    "Waste Collection": [
        "{loc} nalli kasa {t} tegiyalla.",
        "Kasa vilebari vaahana {loc} ge baralilla {t}, kasa raasi aagide.",
        "{lm} hattira kasada dabba tumbi hariyuttide, tumba vaasane, {loc}.",
        "Raste pakka kasa haakidare {lm} hattira {loc}, nayigalu ellaa kade.",
        "{loc} nalli kasa sudtidare, tumba hogeyide.",
        "Mane mane kasa sangraha illa {t} namma street nalli, {loc}.",
        "{lm} hattira satta praani mattu kasa kuliyutide, {loc}.",
        "{loc} nalli kasada raasi tegiyiri, sanjigalu hechchaguttive.",
        "Ola kasa mattu odd kasa mix maadtidaare {loc} nalli.",
        "Chaduru hattira kasa tumbide, {lm} hattira {loc}.",
    ],
}
KA = {
    "Street Lighting": [
        "{loc} ಬಳಿ ಬೀದಿ ದೀಪ {t} ಉರಿಯುತ್ತಿಲ್ಲ.",
        "ನಮ್ಮ ರಸ್ತೆಯಲ್ಲಿ ಬೀದಿ ದೀಪಗಳು ಹಾಳಾಗಿವೆ, {loc} ಪ್ರದೇಶದಲ್ಲಿ ಎಲ್ಲೆಡೆ ಕತ್ತಲೆ.",
        "{loc} ನಲ್ಲಿ street light {t} ಕೆಲಸ ಮಾಡುತ್ತಿಲ್ಲ.",
    ],
    "Water Supply": [
        "{loc} ನಲ್ಲಿ {t} ನೀರು ಬರುತ್ತಿಲ್ಲ.",
        "ನಮ್ಮ ಏರಿಯಾ {loc} ನಲ್ಲಿ ಕುಡಿಯುವ ನೀರು ಕೊಳಕಾಗಿದೆ.",
        "{loc} ಹತ್ತಿರ ಪೈಪ್ ಒಡೆದು ನೀರು ಸೋರುತ್ತಿದೆ.",
    ],
    "Roads and Potholes": [
        "{loc} ಹತ್ತಿರ ರಸ್ತೆಯಲ್ಲಿ ದೊಡ್ಡ ಗುಂಡಿ ಇದೆ.",
        "{loc} ನಲ್ಲಿ ರಸ್ತೆ ಹಾಳಾಗಿದೆ, ಗುಂಡಿಗಳು {t} ಹಾಗೆಯೇ ಇವೆ.",
        "{loc} ರಸ್ತೆಯಲ್ಲಿ ಮ್ಯಾನ್‌ಹೋಲ್ ಮುಚ್ಚಳ ಇಲ್ಲ.",
    ],
    "Waste Collection": [
        "{loc} ನಲ್ಲಿ ಕಸ {t} ತೆಗೆಯುತ್ತಿಲ್ಲ.",
        "{loc} ಹತ್ತಿರ ಕಸದ ರಾಶಿ ಬಿದ್ದಿದೆ, ತುಂಬಾ ವಾಸನೆ.",
        "ಕಸದ ಗಾಡಿ ನಮ್ಮ ಬೀದಿಗೆ ಬರುತ್ತಿಲ್ಲ, {loc}.",
    ],
}

# Hard / ambiguous cases ("light", "cut", "pipe", "drain", "dump") – human-written style
HARD = [
    ("Street Lighting", "The light near {loc} is out. It is a street light not a traffic light.", "Medium", "Negative"),
    ("Street Lighting", "Light cut hogide {loc} nalli, street light off.", "Medium", "Negative"),
    ("Water Supply", "Water cut at {loc}, taps dry {t}.", "High", "Negative"),
    ("Water Supply", "Pipe cut by the cable contractors near {lm} in {loc}, no water.", "High", "Negative"),
    ("Roads and Potholes", "Road cut not repaired near {lm} in {loc}, please fix.", "Medium", "Neutral"),
    ("Roads and Potholes", "Drain cover broken on the road at {loc}, a deep hole.", "High", "Negative"),
    ("Waste Collection", "Garbage dump near the drain at {loc}, please clean.", "Medium", "Neutral"),
    ("Waste Collection", "Tree cut waste and branches left on the road at {loc} {t}.", "Low", "Neutral"),
]

# ---- noise ------------------------------------------------------------------------------
SMS = {
    "please": "plz", "pls": "plz", "street": "stret", "light": "lite", "lights": "lites",
    "water": "watr", "garbage": "garbge", "pothole": "pothol", "potholes": "potholes",
    "road": "rd", "not": "nt", "since": "sinc", "urgent": "urgnt", "urgently": "urgntly",
    "supply": "suply", "collected": "colected", "working": "workin", "you": "u", "your": "ur",
    "with": "wid", "the": "d", "yesterday": "yday", "tomorrow": "tmrw",
}


def add_noise(text, level):
    if level == 0:
        return text
    words = text.split(" ")
    out = []
    for w in words:
        core = w.lower().strip(".,!?")
        if random.random() < 0.28 and core in SMS:
            w = SMS[core] + (w[-1] if w[-1] in ".,!?" else "")
        elif random.random() < 0.07 and len(core) > 4 and core.isascii():
            i = random.randint(1, len(w) - 3)           # swap two inner characters
            w = w[:i] + w[i + 1] + w[i] + w[i + 2:]
        elif random.random() < 0.04 and len(core) > 3 and core.isascii():
            i = random.randint(1, len(w) - 2)            # drop a character
            w = w[:i] + w[i + 1:]
        out.append(w)
    t = " ".join(out)
    if level >= 2:
        t = t.lower()
        t = t.replace(".", "").replace(",", "") if random.random() < 0.6 else t
    if level >= 3 and random.random() < 0.5:
        t = t + random.choice([" pls", " plz", " !!", " asap", " ..."])
    return t


def fill(tmpl, loc, lm, t):
    return tmpl.format(loc=loc, lm=lm, t=t)


# Sentiment cues: Negative complaints often vent; Neutral ones are polite/informational.
SENT_PH = {
    "english": {
        "Negative": ["This is very frustrating.", "Totally unacceptable!", "We are fed up with this negligence.",
                     "Worst service, nobody cares.", "Shame on the department.", "Very disappointed."],
        "Neutral": ["Thank you.", "Thanks in advance.", "Kindly acknowledge.", "Regards.", "Grateful for your help."],
    },
    "kanglish": {
        "Negative": ["tumba kasta aagide.", "yaarigu kaalji illa, bejaar aagide.", "nimma department yenu maaduttide?",
                     "idu sahisoke aagalla.", "tumba kopa barutte."],
        "Neutral": ["dhanyavadagalu.", "thanks.", "dhanyavaada.", "gamanisi, thanks."],
    },
    "kannada_script": {
        "Negative": ["ತುಂಬಾ ಕಷ್ಟವಾಗಿದೆ.", "ಇದು ಸಹಿಸಲು ಆಗುವುದಿಲ್ಲ.", "ಯಾರಿಗೂ ಕಾಳಜಿ ಇಲ್ಲ."],
        "Neutral": ["ಧನ್ಯವಾದಗಳು.", "ಧನ್ಯವಾದ."],
    },
}

SENT_BY_URG = {"High": ["Negative"] * 9 + ["Neutral"], "Medium": ["Negative"] * 5 + ["Neutral"] * 5,
               "Low": ["Neutral"] * 8 + ["Negative"] * 2}
URG_WEIGHTS = {"High": 0.3, "Medium": 0.45, "Low": 0.25}


def make_row(i, style, cat):
    loc = random.choice(LOCATIONS)
    lm = random.choice(LANDMARKS)
    urg = random.choices(list(URG_WEIGHTS), weights=URG_WEIGHTS.values())[0]
    use_time = random.random() < 0.65
    if style == "english":
        t = random.choice(TIMES_EN) if use_time else ""
        body = EN[cat]
        urg_ph = URG_EN
    elif style == "kanglish":
        t = random.choice(TIMES_KN) if use_time else ""
        body = KN[cat]
        urg_ph = URG_KN
    else:
        t = random.choice(TIMES_KA) if use_time else ""
        body = KA[cat]
        urg_ph = URG_KA
    tmpl = random.choice(body)
    if not use_time or "{t}" not in tmpl:
        t = ""  # gold time label must match what actually appears in the text
        tmpl = tmpl.replace(" {t}", "").replace("{t} ", "")
    text = fill(tmpl, loc, lm, t).replace("  ", " ").replace(" ,", ",")
    # Some English complaints are code-mixed in the closing line (and vice versa)
    if style == "english" and random.random() < 0.2:
        text += " " + random.choice(URG_KN[urg])
    else:
        text += " " + random.choice(urg_ph[urg])
    sent = random.choice(SENT_BY_URG[urg])
    if random.random() < 0.75:  # 25% of rows carry no explicit sentiment cue (realistic noise)
        text += " " + random.choice(SENT_PH[style][sent])
    if style != "kannada_script":
        text = add_noise(text, random.choice([0, 1, 1, 2, 3]))
    return dict(id=i, text=text.strip(), category=cat, urgency=urg, sentiment=sent,
                lang_style=style, location=loc, time=t)


def main():
    rows = []
    i = 1
    per_cat = {"english": 85, "kanglish": 105, "kannada_script": 20}
    for cat in CATEGORIES:
        for style, n in per_cat.items():
            for _ in range(n):
                rows.append(make_row(i, style, cat))
                i += 1
    for _ in range(4):  # hard / ambiguous cases, repeated with different slots
        for cat, tmpl, urg, sent in HARD:
            loc, lm = random.choice(LOCATIONS), random.choice(LANDMARKS)
            t = random.choice(TIMES_EN)
            rows.append(dict(id=i, text=add_noise(fill(tmpl, loc, lm, t), random.choice([0, 1])),
                             category=cat, urgency=urg, sentiment=sent, lang_style="hard",
                             location=loc, time=t))
            i += 1
    random.shuffle(rows)
    for n, r in enumerate(rows, 1):
        r["id"] = n
    with open("data/complaints.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} complaints to data/complaints.csv")


if __name__ == "__main__":
    main()
