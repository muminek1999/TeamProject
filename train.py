import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
import numpy as np
from sklearn.pipeline import Pipeline
from joblib import dump

from functions import extract_features, log1p_clip

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

    num_cols = [
        'url_len', 'domain_len', 'subdomain_count', 'ip_len', 'ip_dot_count',
        'js_len', 'js_obf_len', 'content_len'
    ]

    log_cols = ['js_len', 'js_obf_len', 'content_len']
    other_num = list(set(num_cols) - set(log_cols))

    cat_cols = ['uses_https', 'who_is_complete', 'tld', 'geo_loc']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), other_num),
        ('cat', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=5, sparse_output=False), cat_cols)])

    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, class_weight='balanced')

    pipe = Pipeline(steps=[('prep', preprocessor),
                           ('clf', clf)])

    y = df['label'].map({'good': 1, 'bad': 0})
    pipe.fit(feat_df, y)
    print("✅  Model trained")

    # plot_feature_importance(...)

    dump(pipe, "model.joblib")
    print("💾  Model saved as model.joblib")