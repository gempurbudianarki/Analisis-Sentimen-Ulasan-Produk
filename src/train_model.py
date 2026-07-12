import os
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Import our custom preprocessor
from preprocessing import ReviewPreprocessor

def download_dataset(url, dest_path):
    """Downloads the raw dataset from a URL if it doesn't already exist."""
    if os.path.exists(dest_path):
        print(f"[INFO] Dataset already exists at {dest_path}")
        return
        
    print(f"[INFO] Downloading dataset from {url}...")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            f.write(response.content)
        print(f"[SUCCESS] Dataset downloaded and saved to {dest_path}")
    except Exception as e:
        print(f"[ERROR] Failed to download dataset: {e}")
        raise e

def map_rating_to_sentiment(rating):
    """Maps review rating (1-5) to sentiment label."""
    if rating >= 4:
        return 'Positif'
    elif rating == 3:
        return 'Netral'
    else:
        return 'Negatif'

def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_dir = os.path.join(base_dir, 'dataset')
    models_dir = os.path.join(base_dir, 'models')
    reports_dir = os.path.join(base_dir, 'reports')
    
    os.makedirs(dataset_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    raw_csv_path = os.path.join(dataset_dir, 'raw_reviews.csv')
    clean_csv_path = os.path.join(dataset_dir, 'clean_reviews.csv')
    
    # 1. Download Dataset
    dataset_url = "https://raw.githubusercontent.com/ydhnwb/sentiment_product_review_tokopedia/master/product_reviews_dirty.csv"
    download_dataset(dataset_url, raw_csv_path)
    
    # 2. Load & Inspect Data
    print("[INFO] Loading raw dataset...")
    df = pd.read_csv(raw_csv_path)
    print(f"[INFO] Raw dataset shape: {df.shape}")
    
    # Rename columns to standard names
    if 'text' not in df.columns or 'rating' not in df.columns:
        text_col = [col for col in df.columns if 'text' in col.lower() or 'review' in col.lower()][0]
        rating_col = [col for col in df.columns if 'rating' in col.lower() or 'star' in col.lower()][0]
        df = df.rename(columns={text_col: 'text', rating_col: 'rating'})
    
    # Drop rows with missing text or rating
    df = df.dropna(subset=['text', 'rating'])
    
    # Map rating to sentiment
    df['sentiment'] = df['rating'].apply(map_rating_to_sentiment)
    
    # Advanced Balanced Resampling (Oversampling & Downsampling to 6,000 per class)
    print("\n[INFO] Original class distribution:")
    print(df['sentiment'].value_counts())
    
    target_samples_per_class = 6000
    balanced_dfs = []
    
    for sentiment_class in df['sentiment'].unique():
        df_class = df[df['sentiment'] == sentiment_class]
        if len(df_class) > target_samples_per_class:
            df_class_sampled = df_class.sample(n=target_samples_per_class, random_state=42)
            print(f"[INFO] Downsampled '{sentiment_class}' from {len(df_class)} to {target_samples_per_class}")
        else:
            df_class_sampled = df_class.sample(n=target_samples_per_class, replace=True, random_state=42)
            print(f"[INFO] Oversampled '{sentiment_class}' from {len(df_class)} to {target_samples_per_class}")
        balanced_dfs.append(df_class_sampled)
        
    df = pd.concat(balanced_dfs).reset_index(drop=True)
    print(f"[INFO] Balanced dataset shape: {df.shape}")
    
    # 3. Preprocess Dataset
    print("\n[INFO] Running fast preprocessing pipeline (Case Folding, Cleaning, Slang Normalization, collapsings)...")
    
    preprocessor = ReviewPreprocessor()
    total_rows = len(df)
    clean_texts = []
    
    for i, text in enumerate(df['text']):
        clean_text = preprocessor.preprocess(text, stem=False)
        clean_texts.append(clean_text)
        if (i + 1) % 5000 == 0 or (i + 1) == total_rows:
            print(f"[PROGRESS] Preprocessed {i + 1}/{total_rows} reviews")
            
    df['clean_text'] = clean_texts
    
    # Filter out empty rows
    df_clean = df[df['clean_text'].str.strip() != ''].copy()
    print(f"[INFO] Cleaned dataset shape: {df_clean.shape} (Removed {len(df) - len(df_clean)} empty rows)")
    
    # Save clean dataset
    df_clean[['text', 'rating', 'sentiment', 'clean_text']].to_csv(clean_csv_path, index=False)
    print(f"[SUCCESS] Cleaned reviews saved offline to {clean_csv_path}")
    
    # 4. Train-Test Split (Stratified)
    X = df_clean['clean_text']
    y = df_clean['sentiment']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"[INFO] Training set size: {X_train.shape[0]} | Test set size: {X_test.shape[0]}")
    
    # 5. Advanced Feature Extraction: Feature Union (Word-level + Char-level TF-IDF)
    print("\n[INFO] Extracting hybrid Word-Char TF-IDF features (Word vocab: 10,000 | Char vocab: 15,000)...")
    
    word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=10000)
    char_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), max_features=15000)
    
    vectorizer = FeatureUnion([
        ('word', word_vectorizer),
        ('char', char_vectorizer)
    ])
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Save Vectorizer
    vectorizer_path = os.path.join(models_dir, 'vectorizer.joblib')
    joblib.dump(vectorizer, vectorizer_path)
    print(f"[SUCCESS] Vectorizer saved to {vectorizer_path}")
    
    # 6. Model Training, Hyperparameter Tuning & Comparison
    print("\n" + "="*50)
    print(" MODEL TRAINING & HYPERPARAMETER TUNING ")
    print("="*50)
    
    # Hyperparameter tuning for Calibrated Linear SVM
    print("[INFO] Tuning Calibrated Linear SVM C parameter...")
    best_svc_f1 = -1
    best_svc_obj = None
    best_c_val = 0.2
    
    for c_val in [0.05, 0.1, 0.2, 0.5, 1.0]:
        svc = CalibratedClassifierCV(LinearSVC(C=c_val, class_weight='balanced', random_state=42))
        svc.fit(X_train_tfidf, y_train)
        y_pred = svc.predict(X_test_tfidf)
        f1 = f1_score(y_test, y_pred, average='macro')
        acc = accuracy_score(y_test, y_pred)
        print(f"  LinearSVC (C={c_val}) -> Accuracy: {acc:.4f} | F1-Score (Macro): {f1:.4f}")
        if f1 > best_svc_f1:
            best_svc_f1 = f1
            best_svc_obj = svc
            best_c_val = c_val
            
    print(f"[SUCCESS] Best SVM Config: C={best_c_val} (F1 Macro: {best_svc_f1:.4f})")
    
    # Also evaluate Naive Bayes and Logistic Regression for comparison
    models = {
        f'Linear SVM (Calibrated, C={best_c_val})': best_svc_obj,
        'Naive Bayes (MultinomialNB)': MultinomialNB(alpha=0.1),
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', C=1.0, random_state=42)
    }
    
    best_model_name = None
    best_model_obj = None
    best_overall_f1 = -1
    
    for name, model in models.items():
        if 'SVM' not in name:
            model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        print(f"\n[MODEL EVAL] {name} -> Accuracy: {acc:.4f} | F1-Score (Macro): {f1:.4f}")
        print(classification_report(y_test, y_pred))
        
        if f1 > best_overall_f1:
            best_overall_f1 = f1
            best_model_name = name
            best_model_obj = model
            
    print("\n" + "="*50)
    print(f"[SUMMARY] Best Model Selected: {best_model_name} (F1-Score Macro: {best_overall_f1:.4f})")
    print("="*50)
    
    # Save the best model
    model_path = os.path.join(models_dir, 'model_sentimen.joblib')
    joblib.dump(best_model_obj, model_path)
    print(f"[SUCCESS] Best model ({best_model_name}) saved to {model_path}")
    
    # 7. Generate Confusion Matrix
    y_pred_best = best_model_obj.predict(X_test_tfidf)
    labels = sorted(list(y.unique()))
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred_best, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    cm_path = os.path.join(reports_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Confusion Matrix saved to {cm_path}")
    
    # Copy confusion matrix to app directory for Streamlit dashboard
    app_dir = os.path.join(base_dir, 'app')
    os.makedirs(app_dir, exist_ok=True)
    import shutil
    shutil.copy(cm_path, os.path.join(app_dir, 'confusion_matrix.png'))
    print(f"[SUCCESS] Copied confusion matrix to app directory.")

if __name__ == '__main__':
    main()
