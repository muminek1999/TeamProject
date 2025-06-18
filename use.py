import pandas as pd
from joblib import load
from functions import extract_features, extract_domain, get_ip, get_whois, get_geolocation, get_js_features

def classify_url(url):
    try:
        model = load("model.joblib")
    except FileNotFoundError:
        print("❌  Error: File model.joblib not found. Run first: 'python <path> train'")
        return

    domain = extract_domain(url)
    ip = get_ip(domain)
    whois = get_whois(domain)
    geo_loc = get_geolocation(ip)
    js_features = get_js_features(url)

    dummy_row = {
        'url': url,
        'https': 'yes' if url.lower().startswith('https') else 'no',
        'who_is': whois,
        'tld': '.' + url.split('.')[-1],
        'ip_add': ip,
        'geo_loc': geo_loc.get('country', 'XX'),
        'content': js_features['content'],
        'js_len': js_features['js_len'],
        'js_obf_len': js_features['js_obf_len'],
    }

    features = extract_features(dummy_row)
    df = pd.DataFrame([features])
    df = pd.get_dummies(df)

    model_features = model.feature_names_in_

    missing_cols = [col for col in model_features if col not in df.columns]
    missing_df = pd.DataFrame(0, index=df.index, columns=missing_cols)
    df = pd.concat([df, missing_df], axis=1)[model_features]

    prediction = model.predict(df)[0]
    print("Benign" if prediction == 1 else "Malicious")
