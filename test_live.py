import pandas as pd
from joblib import load
from functions import extract_features


def test_model_live(csv_path, n):
    try:
        model = load("model.joblib")
    except FileNotFoundError:
        print("❌  File model.joblib not found. First run: python main.py train <csv_path>")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"❌  Error loading CSV file: {e}")
        return

    if n > len(df):
        print(f"⚠️  Requested number of samples ({n}) exceeds dataset size ({len(df)}). Using all available data")
        n = len(df)

    n_bad = int(n * 0.2)
    n_good = n - n_bad

    df_bad = df[df['label'] == 'bad'].sample(n=n_bad, replace=True)
    df_good = df[df['label'] == 'good'].sample(n=n_good, replace=False)

    df_sample = pd.concat([df_bad, df_good]).sample(frac=1)

    feat_df = df_sample.apply(extract_features, axis=1, result_type='expand')
    feat_df = pd.get_dummies(feat_df, columns=['tld', 'geo_loc'], prefix=['tld', 'geo'])

    y_true = df_sample['label'].map({'good': 1, 'bad': 0})

    model_features = model.feature_names_in_
    missing_cols = [col for col in model_features if col not in feat_df.columns]
    missing_df = pd.DataFrame(0, index=feat_df.index, columns=missing_cols)
    feat_df = pd.concat([feat_df, missing_df], axis=1)
    X = feat_df[model_features]

    y_pred = model.predict(X)
    correct = (y_true == y_pred).sum()
    print(f"✅  Correct predictions: {correct} / {n}")