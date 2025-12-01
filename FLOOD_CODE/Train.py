"""
flood_detection_training.py
Full end-to-end script to:
- Load dataset (default path: /mnt/data/flood_dataset.csv)
- Do EDA and save graphs (histograms, boxplots, correlation heatmap)
- Train 5 ML models (LogisticRegression, RandomForest, SVC, GradientBoosting, KNeighbors)
- Save confusion matrix for each model and a comparison bar plot of evaluation metrics
- Store the best model (by balanced accuracy / f1) to disk
- Provide a standalone `predict_single_sample()` function that loads the saved model+scaler

Usage:
    python flood_detection_training.py --data /path/to/flood_dataset.csv

Outputs (in the current working directory):
    ./outputs/eda_*.png
    ./outputs/confusion_{modelname}.png
    ./outputs/comparison_metrics.png
    ./outputs/best_model.joblib
    ./outputs/scaler.joblib

Requirements:
    pip install pandas numpy matplotlib scikit-learn seaborn joblib

"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
import joblib

# ------------------------- Utilities -------------------------

def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)


# ------------------------- EDA -------------------------

def run_eda(df, outdir):
    ensure_dir(outdir)
    features = ['water_level', 'rain', 'soil_moisture']

    # Histograms
    for col in features:
        plt.figure(figsize=(6,4))
        plt.hist(df[col], bins=40, edgecolor='k', alpha=0.7)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('count')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'eda_hist_{col}.png'))
        plt.close()

    # Boxplots grouped by target
    for col in features:
        plt.figure(figsize=(6,4))
        sns.boxplot(x='target', y=col, data=df)
        plt.title(f'Boxplot of {col} by target')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'eda_box_{col}.png'))
        plt.close()

    # Correlation heatmap
    plt.figure(figsize=(6,5))
    corr = df[features].corr()
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Feature correlation heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'eda_correlation_heatmap.png'))
    plt.close()

    # Pairplot - can be heavy, optional
    try:
        sns.pairplot(df[features + ['target']], hue='target', corner=True)
        plt.savefig(os.path.join(outdir, 'eda_pairplot.png'))
        plt.close()
    except Exception as e:
        print('pairplot skipped (heavy):', e)


# ------------------------- Modeling -------------------------

def plot_and_save_confusion(y_true, y_pred, label_encoder, outpath, title='Confusion Matrix'):
    """
    y_true, y_pred : integer-encoded labels (e.g. 0,1,2)
    label_encoder   : fitted sklearn.preprocessing.LabelEncoder (so we can get string names)
    outpath         : path to save the PNG
    """
    # numeric labels present in y_true / y_pred
    labels_int = np.unique(np.concatenate([y_true, y_pred]))
    # For consistent ordering, use range(len(classes)) if you want full matrix even if some classes missing:
    full_labels_int = np.arange(len(label_encoder.classes_))

    # Compute confusion matrix for the full set so each class appears in matrix (rows/cols)
    cm = confusion_matrix(y_true, y_pred, labels=full_labels_int)

    # Convert integer labels to string names for display
    class_names = list(label_encoder.classes_)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


# --- Update train_and_compare to call the new function ---
def train_and_compare(X_train, X_test, y_train, y_test, label_encoder, outdir):
    ensure_dir(outdir)

    models = {
        'logistic': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=200, random_state=42),
        'svc': SVC(probability=True, random_state=42),
        'gradient_boost': GradientBoostingClassifier(n_estimators=200, random_state=42),
        'knn': KNeighborsClassifier(n_neighbors=7)
    }

    results = []

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted')

        print(f"{name} -> accuracy: {acc:.4f}, f1: {f1:.4f}")
        print(classification_report(y_test, y_pred, target_names=list(label_encoder.classes_)))

        # Save confusion matrix (pass the fitted LabelEncoder)
        plot_and_save_confusion(y_test, y_pred, label_encoder,
                                 outpath=os.path.join(outdir, f'confusion_{name}.png'),
                                 title=f'Confusion Matrix - {name}')

        # Save model
        joblib.dump(model, os.path.join(outdir, f'model_{name}.joblib'))

        results.append({
            'model': name,
            'accuracy': acc,
            'f1_weighted': f1,
            'precision_weighted': prec,
            'recall_weighted': rec
        })

    results_df = pd.DataFrame(results).sort_values(by='f1_weighted', ascending=False)
    results_df.to_csv(os.path.join(outdir, 'model_comparison_metrics.csv'), index=False)

    # Bar plot comparison
    plt.figure(figsize=(8,5))
    x = np.arange(len(results_df))
    plt.bar(x - 0.2, results_df['accuracy'], width=0.4, label='accuracy')
    plt.bar(x + 0.2, results_df['f1_weighted'], width=0.4, label='f1_weighted')
    plt.xticks(x, results_df['model'])
    plt.ylabel('Score')
    plt.ylim(0,1)
    plt.legend()
    plt.title('Model comparison: accuracy vs f1_weighted')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'comparison_metrics.png'))
    plt.close()

    return results_df


# ------------------------- Prediction helper -------------------------

def predict_single_sample(sample, model_path, scaler_path=None, label_encoder_path=None):
    """
    sample: dict with keys 'water_level', 'rain', 'soil_moisture'
    model_path: path to saved model (joblib)
    scaler_path: optional path to saved scaler (joblib) if you used scaling
    label_encoder_path: optional path to saved LabelEncoder
    Returns: predicted label and probabilities (if available)
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if scaler_path and os.path.exists(scaler_path) else None
    le = joblib.load(label_encoder_path) if label_encoder_path and os.path.exists(label_encoder_path) else None

    X = np.array([[sample['water_level'], sample['rain'], sample['soil_moisture']]], dtype=float)
    if scaler is not None:
        X = scaler.transform(X)

    pred = model.predict(X)
    proba = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)[0]

    if le is not None:
        pred_label = le.inverse_transform(pred)[0]
        proba_dict = None
        if proba is not None:
            proba_dict = {cls: float(p) for cls, p in zip(le.classes_, proba)}
            
        return pred_label, proba_dict
    else:
        return pred[0], (list(map(float, proba)) if proba is not None else None)


# ------------------------- Main script -------------------------

def main(args):
    df = pd.read_csv(args.data)
    print('Loaded', len(df), 'rows')

    # Basic cleaning / checks
    expected_cols = {'water_level', 'rain', 'soil_moisture', 'target'}
    if not expected_cols.issubset(set(df.columns)):
        raise ValueError(f"CSV must contain columns: {expected_cols}")

    # Run EDA
    print('Running EDA...')
    run_eda(df, args.outdir)

    # Encode target
    le = LabelEncoder()
    df['target_enc'] = le.fit_transform(df['target'])
    joblib.dump(le, os.path.join(args.outdir, 'label_encoder.joblib'))

    X = df[['water_level', 'rain', 'soil_moisture']].values
    y = df['target_enc'].values

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Scaling (important for SVC, KNN, Logistic)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(args.outdir, 'scaler.joblib'))

    # We'll train two versions of models: ones that take scaled data (logistic, svc, knn) and tree-based on raw
    # For simplicity we will feed scaled data into all models — tree-based methods are robust to scaling.

    # results_df = train_and_compare(X_train_scaled, X_test_scaled, y_train, y_test, labels=list(le.classes_), outdir=args.outdir)
    results_df = train_and_compare(X_train_scaled, X_test_scaled, y_train, y_test, label_encoder=le, outdir=args.outdir)

    # Save best model
    best_model_name = results_df.iloc[0]['model']
    print('Best model by f1_weighted:', best_model_name)

    src_model_path = os.path.join(args.outdir, f'model_{best_model_name}.joblib')
    best_model_path = os.path.join(args.outdir, 'best_model.joblib')
    joblib.dump(joblib.load(src_model_path), best_model_path)

    print('Saved best model to', best_model_path)
    print('Scaler and label encoder saved too. Use predict_single_sample() to load and predict a single sample.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train multiple ML models for Flood Detection')
    parser.add_argument('--data', type=str, default='data.csv', help='Path to CSV dataset')
    parser.add_argument('--outdir', type=str, default='outputs', help='Directory to save outputs')
    args = parser.parse_args()

    main(args)

# ------------------------- Example: standalone prediction usage -------------------------
#
# After running the training script, in another script (or interactive shell) you can do:
#
# from joblib import load
# from flood_detection_training import predict_single_sample
#
# sample = {'water_level': 4.5, 'rain': 3800, 'soil_moisture': 3700}
# pred_label, proba = predict_single_sample(sample,
#                                          model_path='./outputs/best_model.joblib',
#                                          scaler_path='./outputs/scaler.joblib',
#                                          label_encoder_path='./outputs/label_encoder.joblib')
# print('Prediction:', pred_label)
# print('Probabilities:', proba)
#
# You can also call the script from command line and then import the predict_single_sample function in a test harness.
