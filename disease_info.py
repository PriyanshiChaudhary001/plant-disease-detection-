"""Precaution / cure knowledge base. The model predicts the NAME; this file supplies the advice."""

INFO = {
    "early blight": {
        "severity": "Medium-High",
        "precaution": "Avoid overhead watering, rotate crops every season, remove infected lower leaves, keep plants spaced for airflow.",
        "cure": "Spray a fungicide such as Mancozeb or Chlorothalonil every 7-10 days; remove and destroy badly infected leaves.",
    },
    "late blight": {
        "severity": "Very High",
        "precaution": "Use disease-free seed/seedlings, avoid waterlogging, do not plant near infected potato/tomato fields.",
        "cure": "Apply Metalaxyl + Mancozeb or copper-based fungicide immediately; uproot and destroy severely infected plants.",
    },
    "leaf mold": {
        "severity": "Medium",
        "precaution": "Reduce humidity, improve ventilation, avoid wetting leaves, use resistant varieties.",
        "cure": "Spray Chlorothalonil or copper fungicide; remove infected leaves.",
    },
    "bacterial spot": {
        "severity": "Medium-High",
        "precaution": "Use certified seed, avoid working in wet fields, rotate crops, disinfect tools.",
        "cure": "Copper-based bactericide sprays; remove infected plant parts.",
    },
    "septoria": {
        "severity": "Medium",
        "precaution": "Mulch the soil, avoid overhead watering, rotate crops, clear plant debris after harvest.",
        "cure": "Fungicide with Chlorothalonil or Mancozeb; remove spotted lower leaves.",
    },
    "target spot": {
        "severity": "Medium",
        "precaution": "Improve airflow, avoid dense planting, remove crop residue.",
        "cure": "Apply Azoxystrobin or Chlorothalonil fungicide as per label.",
    },
    "mosaic virus": {
        "severity": "High",
        "precaution": "Control aphids/whiteflies, wash hands and tools, use virus-free seed and resistant varieties.",
        "cure": "No chemical cure for viruses. Remove and destroy infected plants to stop spread.",
    },
    "yellow leaf curl": {
        "severity": "High",
        "precaution": "Control whiteflies with yellow sticky traps, use insect nets, use resistant varieties.",
        "cure": "No direct cure. Remove infected plants; spray neem oil or recommended insecticide against whiteflies.",
    },
    "spider mites": {
        "severity": "Medium",
        "precaution": "Keep plants well watered (mites love dry dust), inspect leaf undersides regularly.",
        "cure": "Spray neem oil, insecticidal soap, or a recommended miticide.",
    },
    "scab": {
        "severity": "Medium",
        "precaution": "Rake and destroy fallen leaves, prune for airflow, choose resistant varieties.",
        "cure": "Fungicide sprays (e.g. Captan or Mancozeb) from early season.",
    },
    "black rot": {
        "severity": "High",
        "precaution": "Prune infected wood, remove mummified fruit, keep canopy open.",
        "cure": "Fungicide such as Myclobutanil or Mancozeb; destroy infected material.",
    },
    "powdery mildew": {
        "severity": "Medium",
        "precaution": "Ensure good airflow and sunlight, avoid excess nitrogen fertilizer.",
        "cure": "Sulphur-based fungicide or neem oil spray.",
    },
    "healthy": {
        "severity": "None",
        "precaution": "Keep up regular watering, balanced fertilizer and routine inspection.",
        "cure": "No treatment needed.",
    },
}

DEFAULT = {
    "severity": "Unknown",
    "precaution": "Isolate the plant and monitor it closely.",
    "cure": "Consult your local agriculture officer / Krishi Vigyan Kendra (KVK) for treatment.",
}


def get_info(class_name):
    name = class_name.lower().replace("_", " ")
    if "healthy" in name:
        return INFO["healthy"]
    for key, val in INFO.items():
        if key in name:
            return val
    return DEFAULT
