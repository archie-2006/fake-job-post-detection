import joblib
from src.data_loader import load_data
from src.class_imbalance import plot_imbalance
from src.preprocessing import create_train_test_split, preprocess_pipeline

def main():
    
    print(f" ==== Starting Fake Job Posting Detection Pipeline ==== ")
    
    print(f"\n ==== Loading data set ==== ")
    df = load_data()

    print(f"\n ==== Analyzing class imbalance ==== ")
    plot_imbalance(df)

    print(f"\n ==== Splitting data ====")
    X_train, X_test, y_train, y_test = create_train_test_split(df)
    
    X_train_processed, X_test_processed, vectorizer, cat_encoder = preprocess_pipeline(X_train, X_test)

    print(f"\n ==== Saving processed data and models ==== ")
    joblib.dump(X_train_processed, 'data/processed/X_train.pkl')
    joblib.dump(X_test_processed, 'data/processed/X_test.pkl')
    joblib.dump(y_train, 'data/processed/y_train.pkl')
    joblib.dump(y_test, 'data/processed/y_test.pkl')
    
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')
    joblib.dump(cat_encoder, 'models/cat_encoder.pkl')

    print(f"\n ==== Final Train Matrix Shape: {X_train_processed.shape} ==== ")

if __name__ == "__main__":
    main()