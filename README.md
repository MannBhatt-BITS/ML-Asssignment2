# CrimeCast: Multiclass Crime Classification

## Problem statement
Predict one of six recorded crime categories from attributes of a historical incident. This is **retrospective categorization**, not prediction of whether or where future crimes will occur.

## Dataset
Source: [Kaggle CrimeCast competition](https://www.kaggle.com/competitions/crime-cast-forecasting-crime-categories). The supplied labeled training CSV has 20,000 records, 21 input columns, and target `Crime_Category`. Class imbalance is substantial. Some investigation fields may not be known when an incident is first reported.

## Approach
- Remove exact duplicates; split once using `random_state=42`, stratified 70% train / 30% test.
- Derive month, weekday, hour, reporting delay, and summary of modus-operandi codes.
- Impute missing values, one-hot encode categorical inputs, and scale numerical features where appropriate. All learned preprocessing is fit **only on the training partition**.
- Exclude direct address/location text, descriptive duplicates, and case status fields. Include incident descriptors only in a retrospective interpretation.
- Train Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, and linear SVM.
- Evaluate on the same untouched labeled holdout: accuracy, macro one-vs-rest ROC AUC, macro precision, macro recall, macro F1, and multiclass MCC.
- For SVM, ROC AUC uses uncalibrated decision scores, which is valid for ranking-based AUC but **not** a probability estimate.

## Local setup
```bash
python -m venv .venv
pip install -r requirements.txt
python train.py --data train.csv
streamlit run app.py
```

`train.csv` is intentionally excluded from Git by `.gitignore`; download it from Kaggle. The training script creates `test_data.csv`, model artifacts under `models/`, and evaluation results under `reports/`. Upload `test_data.csv` in Streamlit for metrics. Kaggle's unlabeled competition test file cannot produce metrics.

## Results
Results on the common stratified holdout (macro metrics; AUC is macro OvR):

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| SVM | 0.8214 | 0.9523 | 0.5830 | 0.6858 | **0.5942** | **0.7093** |
| Logistic Regression | 0.7949 | 0.9487 | 0.5653 | **0.7260** | 0.5763 | 0.6781 |
| Decision Tree | 0.7757 | 0.8897 | 0.5248 | 0.6806 | 0.5574 | 0.6521 |
| KNN | 0.8014 | 0.8610 | **0.6098** | 0.4590 | 0.4756 | 0.6508 |
| Gaussian Naive Bayes | 0.4444 | 0.8339 | 0.4980 | 0.4966 | 0.3228 | 0.3286 |

### Observations
- **SVM:** best macro F1, accuracy, AUC, and MCC; strongest overall baseline under the selected criteria.
- **Logistic Regression:** best macro recall; useful alternative when identifying minority classes is important.
- **Decision Tree:** interpretable nonlinear baseline, but below SVM and Logistic Regression on macro F1.
- **KNN:** best macro precision but notably lower macro recall, consistent with difficulty identifying rare categories.
- **Gaussian Naive Bayes:** lowest accuracy and macro F1; its independence/distribution assumptions may not fit these correlated features.

**Winner:** linear SVM by macro F1. Results depend on the split and feature engineering and should not be presented as a prospective crime forecast.

## GitHub repository
https://github.com/MannBhatt-BITS/ML-Asssignment2

## Live Streamlit app
https://mann-bhatt-ml-assignment2.streamlit.app/

## Limitations
The dataset is imbalanced, rare classes have few examples, and some descriptors are recorded after investigation. Results do not establish causal factors or prospective predictive utility. Random splitting may overestimate generalization to later time periods.

## Academic integrity
Understand, modify, and document your work; preserve a genuine commit history. Include your own Virtual Lab screenshot in the final PDF.
