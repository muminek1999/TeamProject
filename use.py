import pandas as pd
from joblib import load
from functions import extract_features, extract_domain, get_ip, get_whois, get_geolocation, get_js_features
import tldextract

def classify_url(url):
    try:
        pipe = load("model.joblib")
    except FileNotFoundError:
        print("❌  model.joblib not found. Run training first.")
        return

    domain = extract_domain(url)
    ip = get_ip(domain)
    whois = get_whois(domain)
    geo_loc = get_geolocation(ip)
    js_features = get_js_features(url)

    extracted = tldextract.extract(url)
    tld = extracted.suffix.lower()

    dummy_row = {
        'url': url,
        'https': 'yes' if url.lower().startswith('https') else 'no',
        'who_is': whois,
        'tld': tld,
        'ip_add': ip,
        'geo_loc': geo_loc.get('country', 'XX'),
        'content': js_features['content'],
        'js_len': js_features['js_len'],
        'js_obf_len': js_features['js_obf_len'],
    }

    features = extract_features(dummy_row)
    df = pd.DataFrame([features])

    proba = pipe.predict_proba(df)[0]
    pred = pipe.predict(df)[0]

    label = "Benign" if pred == 1 else "Malicious"
    print(f"\n{label}\n(probability = {proba[pred]:.2f})\n")

    return pred, proba[pred]
