import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from train import evaluate

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title='CrimeCast Classification',layout='wide')
st.title('CrimeCast — Multiclass Crime Classification')
st.caption('Five baseline classifiers | fixed stratified holdout | educational demonstration')
metrics_file=ROOT/'reports'/'metrics.csv'
if metrics_file.exists():
    st.subheader('Held-out model comparison')
    st.dataframe(pd.read_csv(metrics_file).set_index('Model').style.format('{:.3f}'),use_container_width=True)
else:
    st.warning('Train models first: python train.py --data train.csv')
model_names=['Logistic Regression','Decision Tree','KNN','Gaussian Naive Bayes','SVM']
selected=st.selectbox('Select classifier',model_names)
uploaded=st.file_uploader('Upload CSV for predictions (optional Crime_Category for evaluation)',type=['csv'])
@st.cache_resource
def load_model(name):
    return joblib.load(ROOT/'models'/(name.lower().replace(' ','_')+'.joblib'))
if uploaded is not None:
    try:
        data=pd.read_csv(uploaded)
        model=load_model(selected)
        predictions=model.predict(data)
        output=data.copy()
        output['Predicted_Crime_Category']=predictions
        st.subheader('Predictions')
        st.dataframe(output.head(100),use_container_width=True)
        st.download_button('Download predictions CSV',output.to_csv(index=False).encode(),'predictions.csv','text/csv')
        if 'Crime_Category' in data.columns:
            valid=data['Crime_Category'].notna()
            labels=list(model.classes_)
            unknown=set(data.loc[valid,'Crime_Category'])-set(labels)
            if unknown:
                st.error(f'Unknown labels: {sorted(unknown)}')
            elif valid.sum()>0:
                y=data.loc[valid,'Crime_Category'].astype(str)
                if set(labels)-set(y):
                    st.info('Some classes are absent in this upload; multiclass AUC may be undefined. Showing report and confusion matrix.')
                else:
                    result,_=evaluate(model,data.loc[valid],y,labels)
                    st.subheader('Uploaded CSV metrics')
                    st.dataframe(pd.DataFrame([result]).style.format('{:.3f}'))
                st.subheader('Classification report')
                st.dataframe(pd.DataFrame(classification_report(y,predictions[valid],labels=labels,output_dict=True,zero_division=0)).T)
                fig,ax=plt.subplots(figsize=(9,7))
                ConfusionMatrixDisplay(confusion_matrix(y,predictions[valid],labels=labels),display_labels=labels).plot(ax=ax,xticks_rotation=90,values_format='d',colorbar=False)
                fig.tight_layout();st.pyplot(fig);plt.close(fig)
    except FileNotFoundError:
        st.error('Model not found. Run train.py and commit model artifacts to the repository.')
    except (KeyError,ValueError) as exc:
        st.error(f'Invalid CSV schema or data: {exc}')
st.info('Retrospective incident categorization only. Not a forecast of future crime or a tool for individual decisions.')
