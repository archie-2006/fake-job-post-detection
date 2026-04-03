import pandas as pd

def load_data(path='data/raw/fake_job_postings.csv'):
    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Target distribution:\n{df['fraudulent'].value_counts(normalize=True)}")
    return df
