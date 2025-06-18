import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import dump

from functions import extract_features, plot_feature_importance

def train_model(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌  File not found: {csv_path}")
        return
    except Exception as e:
        print(f"❌  Error loading CSV file: {e}")
        return

    feat_df = df.apply(extract_features, axis=1, result_type='expand')
    feat_df = pd.get_dummies(feat_df, columns=['tld', 'geo_loc'], prefix=['tld', 'geo'])

    X = feat_df
    y = df['label'].map({'good': 1, 'bad': 0})

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    print("✅  Model trained on the full dataset")

    filtered_features = [feat for feat in ['url_len', 'domain_len', 'subdomain_count', 'uses_https',
                                           'who_is_complete', 'ip_len', 'ip_dot_count']
                         if feat in X.columns]
    indices = [X.columns.get_loc(feat) for feat in filtered_features]
    coefs_orig = model.coef_[0][indices]
    plot_feature_importance(coefs_orig, filtered_features)

    dump(model, "model.joblib")
    print("💾 Model saved as model.joblib")
