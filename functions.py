from urllib.parse import urlparse
import matplotlib.pyplot as plt
import pandas as pd
import socket
import whois
import requests
import tldextract
from bs4 import BeautifulSoup
import urllib.parse
import numpy as np

def extract_features(row):
    url = row['url']
    extracted = tldextract.extract(url)
    tld = extracted.suffix.lower()
    parsed = urlparse(url)
    https_flag = 1 if str(row['https']).lower() in ('yes', 'https', '1', 'true') else 0
    subdomain_count = parsed.netloc.count('.') - 1 if parsed.netloc.count('.') > 0 else 0
    return {
        'url_len': len(url),
        'domain_len': len(parsed.netloc),
        'subdomain_count': subdomain_count,
        'uses_https': https_flag,
        'who_is_complete': 1 if str(row['who_is']).lower() == 'complete' else 0,
        'tld': tld,
        'ip_len': len(str(row['ip_add'])),
        'ip_dot_count': str(row['ip_add']).count('.'),
        'geo_loc': str(row['geo_loc']),

        # 'js_len': float(row['js_len']),
        # 'js_obf_len': float(row['js_obf_len']),
        # 'content_len': len(row['content']) if isinstance(row['content'], str) else 0
    }

def plot_confusion_matrix(cm, labels):
    plt.figure(figsize=(4,4))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = range(len(labels))
    plt.xticks(ticks, labels, rotation=45)
    plt.yticks(ticks, labels)
    thresh = cm.max() / 2
    for i in ticks:
        for j in ticks:
            plt.text(j, i, cm[i, j], ha="center", color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.show()

def plot_feature_importance(coefs, names):
    imp = pd.Series(coefs, index=names).abs().sort_values(ascending=False)
    plt.figure(figsize=(6,4))
    plt.bar(imp.index, imp.values)
    plt.xticks(rotation=45, ha='right')
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()

def extract_domain(url):
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}"

def get_ip(domain):
    return socket.gethostbyname(domain)

def get_whois(domain):
    try:
        w = whois.whois(domain)
        return 'complete' if w.domain_name else 'incomplete'
    except Exception:
        return 'incomplete'

def get_geolocation(ip):
    try:
        response = requests.get(f"https://ipinfo.io/{ip}/json", timeout=5)
        return response.json()
    except Exception:
        return {}

def get_js_features(url):
    try:
        response = requests.get(url, timeout=10)
        html = response.text
        content = html
    except:
        return {
            "js_len": 0,
            "js_obf_len": 0,
            "content_len": 0
        }

    soup = BeautifulSoup(html, "html.parser")
    scripts = soup.find_all("script")

    inline_js = [s.get_text() for s in scripts if s.string]
    inline_js_code = "\n".join(inline_js)

    external_js_code = ""
    for s in scripts:
        if s.has_attr("src"):
            js_url = urllib.parse.urljoin(url, s["src"])
            try:
                js_content = requests.get(js_url, timeout=5).text
                external_js_code += "\n" + js_content
            except:
                continue

    all_js = inline_js_code + external_js_code
    total_len = len(all_js)

    long_lines = [line for line in all_js.splitlines() if len(line) > 200]
    obf_len = sum(len(line) for line in long_lines)

    return {
        "js_len": total_len,
        "js_obf_len": obf_len,
        "content": content
    }

def log1p_clip(arr, max_bytes=1_000_000):
    return np.log1p(np.clip(arr, 0, max_bytes))