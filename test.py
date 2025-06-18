import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from joblib import load
from functions import extract_features, plot_confusion_matrix


def test_model(csv_path):
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

    feat_df = df.apply(extract_features, axis=1, result_type='expand')
    feat_df = pd.get_dummies(feat_df, columns=['tld', 'geo_loc'], prefix=['tld', 'geo'])

    X = feat_df
    y_true = df['label'].map({'good': 1, 'bad': 0})

    # Dopasuj cechy do modelu
    model_features = model.feature_names_in_
    missing_cols = [col for col in model_features if col not in X.columns]
    missing_df = pd.DataFrame(0, index=X.index, columns=missing_cols)
    X = pd.concat([X, missing_df], axis=1)[model_features]

    y_pred = model.predict(X)

    print(f"✅  Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)
    plot_confusion_matrix(cm, ['Malicious (0)', 'Benign (1)'])
