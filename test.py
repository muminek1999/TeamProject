from joblib import load
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from functions import extract_features, plot_confusion_matrix

def test_model(csv_path):
    pipe = load("model.joblib")
    df  = pd.read_csv(csv_path)
    X   = df.apply(extract_features, axis=1, result_type='expand')
    y   = df['label'].map({'good':1,'bad':0})

    y_pred = pipe.predict(X)
    print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
    cm = confusion_matrix(y, y_pred)
    plot_confusion_matrix(cm, ["Malicious","Benign"])
