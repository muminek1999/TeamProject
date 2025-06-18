from urllib.parse import urlparse
import matplotlib.pyplot as plt
import pandas as pd
import socket
import whois
import requests
import tldextract

def extract_features(row):
    url = row['url']
    parsed = urlparse(url)
    https_flag = 1 if str(row['https']).lower() in ('yes', 'https', '1', 'true') else 0
    return {
        'url_len': len(url),
        'domain_len': len(parsed.netloc),
        'subdomain_count': parsed.netloc.count('.'),
        'uses_https': https_flag,
        'who_is_complete': 1 if str(row['who_is']).lower() == 'complete' else 0,
        'tld': str(row['tld']).lstrip('.').lower(),
        'ip_len': len(str(row['ip_add'])),
        'ip_dot_count': str(row['ip_add']).count('.'),
        'geo_loc': str(row['geo_loc'])
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
        return whois.whois(domain)
    except Exception:
        return {}

def get_geolocation(ip):
    try:
        response = requests.get(f"https://ipinfo.io/{ip}/json", timeout=5)
        return response.json()
    except Exception:
        return {}