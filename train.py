"""Train five baseline classifiers on a common stratified holdout.
Run: python train.py --data train.csv
"""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, matthews_corrcoef, roc_auc_score, classification_report, confusion_matrix
from features import CrimeFeatures, NUMERIC, CATEGORICAL

ROOT = Path(__file__).resolve().parent
MODELS = ROOT/'models'

def preprocessor(scale=True):
    nums = [('imputer', SimpleImputer(strategy='median'))]
    if scale:
        nums.append(('scaler', StandardScaler()))
    return ColumnTransformer([
        ('numeric', Pipeline(nums), NUMERIC),
        ('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', min_frequency=10, sparse_output=False))
        ]), CATEGORICAL)
    ], sparse_threshold=0)

def evaluate(model, X, y, labels):
    pred = model.predict(X)
    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, labels=labels, average='macro', zero_division=0)
    if hasattr(model, 'predict_proba'):
        scores = model.predict_proba(X)
    else:
        scores = model.decision_function(X)
    # Binary one-vs-rest AUC accepts either probabilities or uncalibrated decision scores.
    aucs = [roc_auc_score((y == label).astype(int), scores[:, i]) for i, label in enumerate(labels)]
    return {'Accuracy': float(accuracy_score(y,pred)), 'AUC_macro_OvR': float(np.mean(aucs)), 'Precision_macro':float(precision), 'Recall_macro':float(recall), 'F1_macro':float(f1), 'MCC':float(matthews_corrcoef(y,pred))}, pred

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',default=str(ROOT/'train.csv'))
    args=parser.parse_args()
    MODELS.mkdir(exist_ok=True)
    (ROOT/'reports').mkdir(exist_ok=True)
    df=pd.read_csv(args.data)
    df=df.dropna(subset=['Crime_Category']).drop_duplicates()
    X=df.drop(columns=['Crime_Category'])
    y=df['Crime_Category'].astype(str)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.30,stratify=y,random_state=42)
    test=X_test.copy(); test['Crime_Category']=y_test
    test.to_csv(ROOT/'test_data.csv',index=False)
    labels=sorted(y.unique())
    estimators={
      'Logistic Regression':LogisticRegression(max_iter=900, class_weight='balanced'),
      'Decision Tree':DecisionTreeClassifier(max_depth=16,min_samples_leaf=10,class_weight='balanced',random_state=42),
      'KNN':KNeighborsClassifier(n_neighbors=11,weights='distance',n_jobs=-1),
      'Gaussian Naive Bayes':GaussianNB(),
      'SVM':LinearSVC(class_weight='balanced',dual=False,max_iter=3000,random_state=42)
    }
    rows=[]
    for name, estimator in estimators.items():
        print('Training:',name,flush=True)
        pipe=Pipeline([('features',CrimeFeatures()),('preprocess',preprocessor(scale=name!='Decision Tree')),('classifier',estimator)])
        pipe.fit(X_train,y_train)
        metrics,pred=evaluate(pipe,X_test,y_test,labels)
        slug=name.lower().replace(' ','_')
        joblib.dump(pipe,MODELS/f'{slug}.joblib',compress=3)
        (ROOT/'reports'/f'{slug}_classification_report.txt').write_text(classification_report(y_test,pred,labels=labels,zero_division=0))
        pd.DataFrame(confusion_matrix(y_test,pred,labels=labels),index=labels,columns=labels).to_csv(ROOT/'reports'/f'{slug}_confusion_matrix.csv')
        rows.append({'Model':name,**metrics})
        print(metrics,flush=True)
    result=pd.DataFrame(rows).sort_values('F1_macro',ascending=False)
    result.to_csv(ROOT/'reports'/'metrics.csv',index=False)
    (ROOT/'reports'/'metadata.json').write_text(json.dumps({'classes':labels,'training_rows':len(X_train),'test_rows':len(X_test),'random_state':42,'evaluation':'macro averaging; OvR AUC calculated from probabilities or decision scores'},indent=2))
    print('\n',result.to_string(index=False))

if __name__=='__main__':main()
