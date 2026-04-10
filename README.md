# Fake Job Posting Detection

## Package Requirements

| Package | Notes |
|---|---|
| pandas | Data loading and manipulation |
| numpy | Numerical operations |
| scikit-learn | Preprocessing, models, and evaluation |
| matplotlib | Plotting |
| seaborn | Statistical visualizations |
| joblib | Saving and loading model artifacts |
| xgboost | XGBoost classifier |
| wordcloud | Word cloud visualization |
| scipy | Sparse matrix operations (installed with scikit-learn) |

Install all at once inside your virtual environment:

```
pip install pandas numpy scikit-learn matplotlib seaborn joblib xgboost wordcloud
```

---

## Run Instructions

### 1. Download the dataset

The notebook fetches the dataset automatically via `wget` in Cell 4:

```
!wget -O data/raw/fake_job_postings.csv https://raw.githubusercontent.com/abbylmm/fake_job_posting/main/data/fake_job_postings.csv
```

If running locally (without `wget`), download the CSV manually from that URL and place it in the project data/raw/ directory as `fake_job_postings.csv`.

### 2. Set up directories

Cell 3 creates the required folder structure and moves the CSV into place:

```
data/raw/
data/processed/
models/
```

No manual action needed — this runs automatically as part of the notebook.

### 3. Run the notebook top to bottom

Open `final.ipynb` in Jupyter Notebook or JupyterLab and run all cells in order (**Kernel → Restart & Run All**).

The notebook is structured as follows:

1. **Data Loading**: downloads and organises the raw CSV
2. **Exploratory Data Analysis**: class imbalance, text length, categorical risk factors, word frequencies
3. **Preprocessing Pipeline**: text cleaning, TF-IDF vectorisation, one-hot encoding of categorical features, saves processed files to `data/processed/`
4. **Modelling**: trains and cross-validates Naive Bayes, Logistic Regression (with hyperparameter tuning), Linear SVM, Kernel SVM, Random Forest, and XGBoost
5. **Evaluation**: ROC/PR curves, confusion matrices, model comparison
6. **Threshold Tuning**: optimises the decision threshold for fraud detection F1

### 4. Notes

- Processed feature matrices are saved to `data/processed/` and model artifacts to `models/` after the preprocessing cell runs. Later cells load from these paths via `joblib.load` — do not skip the preprocessing cell.
- The Kernel SVM (RBF) cell trains on a 5,000-sample subset by design due to computational cost; this is expected behaviour.
- All other cells are designed to run on the full dataset.