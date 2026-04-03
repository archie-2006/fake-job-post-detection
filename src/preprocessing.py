# src/preprocessing.py
import re
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, csr_matrix

# Clean raw text by lowercasing, removing HTML, punctuation, and normalizing whitespace.
def clean_text(text):
    
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-z\s]', ' ', text)  # Keep only letters and spaces
    text = re.sub(r'\s+', ' ', text).strip()  # Collapse whitespace
    return text

# Splits the dataframe while maintaining the severe 5% fraud class imbalance.
def create_train_test_split(df):

    X = df.drop('fraudulent', axis=1)
    y = df['fraudulent']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.30, 
        stratify=y, 
        random_state=42
    )
    return X_train, X_test, y_train, y_test

# Runs the full text and categorical preprocessing pipeline.
# Fits ONLY on train to prevent data leakage.
def preprocess_pipeline(X_train, X_test):
    import time # Just in case it's not at the top
    
    print("\n[ ~ ] Initializing Preprocessing Pipeline...")
    
    # 1. Combine text columns
    text_cols = ['title', 'description', 'requirements', 'company_profile']
    print(f"\n[ 1 / 4 ] Cleaning {len(text_cols)} text columns...")
    t0 = time.time()
    X_train_text = X_train[text_cols].fillna('').agg(' '.join, axis=1).apply(clean_text)
    X_test_text = X_test[text_cols].fillna('').agg(' '.join, axis=1).apply(clean_text)
    print(f"  === Done in {time.time() - t0:.2f}s")
    print(f"  === Processed {len(X_train_text)} training rows & {len(X_test_text)} testing rows.")

    # 2. TF-IDF Vectorization
    print("\n[ 2 / 4 ] Applying TF-IDF Vectorization...")
    t0 = time.time()
    vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=5,
        max_df=0.7,
        ngram_range=(1, 2),
        stop_words='english',
        dtype=np.float32
    )
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)
    print(f"  === Done in {time.time() - t0:.2f}s")
    print(f"  === Extracted {X_train_tfidf.shape[1]} n-gram features.")

    # 3. Categorical Encoding
    print("\n[ 3 / 4 ] Encoding categorical features...")
    t0 = time.time()
    cat_cols = ['location', 'department', 'employment_type', 'required_experience', 'industry']
    
    X_train_cat_df = X_train[cat_cols].fillna('Unknown')
    X_test_cat_df = X_test[cat_cols].fillna('Unknown')

    cat_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float32)
    X_train_cat = cat_encoder.fit_transform(X_train_cat_df)
    X_test_cat = cat_encoder.transform(X_test_cat_df)
    print(f"  === Done in {time.time() - t0:.2f}s")
    print(f"  === Expanded {len(cat_cols)} categorical columns into {X_train_cat.shape[1]} one-hot features.")

    # 4. Combine Features
    print("\n[ 4 / 4 ] Stacking features into final matrices...")
    t0 = time.time()
    X_train_processed = hstack([X_train_tfidf, X_train_cat]).tocsr()
    X_test_processed = hstack([X_test_tfidf, X_test_cat]).tocsr()
    
    # Calculate sparsity (How much of the matrix is actually zeros)
    train_sparsity = 100 * (1.0 - X_train_processed.nnz / (X_train_processed.shape[0] * X_train_processed.shape[1]))
    
    print(f"  === Done in {time.time() - t0:.2f}s")
    print(f"\n[ ✓ ] Pipeline Complete!")
    print(f"  === Final Train Shape: {X_train_processed.shape}")
    print(f"  === Final Test Shape:  {X_test_processed.shape}")
    print(f"  === Matrix Sparsity:   {train_sparsity:.2f}%")

    return X_train_processed, X_test_processed, vectorizer, cat_encoder