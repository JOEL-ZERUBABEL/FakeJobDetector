import streamlit as st
import pandas as pd
import re
import joblib
import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer

class Frontend:
    def __init__(self):
        self.sentence=joblib.load('sentence_transformer_job_detection.pkl')
        self.regression=joblib.load('logistic_regression_sentence.pkl')


    def upload(self):
        st.header('Job or not')
        st.title('Upload a job posting information passage to check whether it is real or not')
        uploading_passage=st.text_input('enter the passage text here')
        if st.button('Check'):
            if uploading_passage.strip()=='':
                st.warning('the box is empty')
            pred,prob=self.predict(uploading_passage)
            label='Real Job Announcement' if pred==0 else 'Fake Job Announcement'
            st.markdown(f'Prediction:{label}')
            self.plot(prob)
    def plot(self,prob):
        chart=pd.DataFrame(
            {'Category':['Genuine','Fake'],'Probablity':prob})
        st.bar_chart(chart.set_index('Category'))
        st.info(f'Prediction Confidence: {np.max(prob)*100:.2}%')

    def clean(self,text):
        clean=str(text).lower()
        clean=re.sub(r'<.*?>','',clean)
        clean=re.sub(r'\d+','',clean)
        words = re.findall(r'\b\w+\b', text)
        words = [w for w in words if w not in ENGLISH_STOP_WORDS]
        return ' '.join(words)
    
    def predict(self,text):
        clean=self.clean(text)
        encode=self.sentence.encode([clean])
        pred=self.regression.predict(encode)[0]
        prob=self.regression.predict_proba(encode)[0]
        return pred,prob

if __name__ == '__main__':
    app = Frontend()
    app.upload()