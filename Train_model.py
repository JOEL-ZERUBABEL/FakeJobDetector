import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
#print(df.coloums)
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
import joblib
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier


class Train_model:
    def __init__(self):
        self.df=pd.read_csv('fake_job_postings.csv')
        print(self.df['fraudulent'].value_counts())

        self.vectorizer=CountVectorizer(stop_words='english',min_df=1,max_df=0.85,token_pattern=r'\b\w+b')
        self.sentence_transformer=SentenceTransformer('all-MiniLM-L6-v2')
        self.tfidtransformer=TfidfVectorizer(stop_words='english',max_features=5000,max_df=0.85)
        self.word2vec=Word2Vec(vector_size=300,epochs=100,workers=4,min_count=2)
        self.naive_bayes=MultinomialNB()
        


    def clean(self,text):
        clean=str(text).lower()
        clean=re.sub(r'<.*?>','',clean)
        clean=re.sub(r'\d+','',clean)
        words = re.findall(r'\b\w+\b', text)
        words = [w for w in words if w not in ENGLISH_STOP_WORDS]
        return ' '.join(words)
    
    def prepare_data(self):        
        self.X=(self.df['title'].fillna('')+' '+self.df['description'].fillna('')).apply(self.clean)
        self.y=self.df['fraudulent']

    def count_tfid(self):
        return self.tfidtransformer.fit_transform(self.X)
       
    
    def count_vectorize(self):
        return self.vectorizer.fit_transform(self.X)
    
    def word2_vectorize(self):
        tokenized=[text.split() for text in self.X]
        self.word2vec.build_vocab(tokenized,progress_per=1000)
        self.word2vec.train(tokenized, total_examples=len(tokenized), epochs=10)
        Xwv=[np.mean([self.word2vec.wv[word] for word in words if word in self.word2vec.wv], axis=0) for words in tokenized]
        return np.array(Xwv)

    def count_sentence(self):
        return self.sentence_transformer.encode(self.X.tolist(),show_progress_bar=True)
    
    def train_data(self):
        #Tfidtransformer
        tfid=self.count_tfid()
        X_train,X_test,y_train,y_test=train_test_split(tfid,self.y,test_size=0.3,random_state=42)
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        self.naive_bayes.fit(X_train_res,y_train_res)
        y_pred=self.naive_bayes.predict(X_test)
        print('Tfid Classification+Naive Bayes')
        print(f'accuracy_score\n{accuracy_score(y_test,y_pred)}')
        print(f'classification report\n{classification_report(y_test,y_pred)}')

        #Word2vec
        word2vec=self.word2_vectorize()
        X_train,X_test,y_train,y_test=train_test_split(word2vec,self.y,test_size=0.3,random_state=42)
        sm = SMOTE(random_state=42)
        X_train_res,y_train_res=sm.fit_resample(X_train, y_train)
        logic=LogisticRegression(max_iter=500,class_weight='balanced')
        logic.fit(X_train_res,y_train_res)
        y_pred=logic.predict(X_test)
        print('Word2vec Classification+Logistic Regression')
        print(f'accuracy_score\n{accuracy_score(y_test,y_pred)}')

        #Sentencetransformer
        sentence=self.count_sentence()
        X_train,X_test,y_train,y_test=train_test_split(sentence,self.y,test_size=0.3,random_state=42)
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res=sm.fit_resample(X_train, y_train)
        logic=LogisticRegression(max_iter=500,class_weight='balanced')
        logic.fit(X_train_res,y_train_res)
        y_pred=logic.predict(X_test)
        self.sentence_classifier = logic
        print('Sentencetransfomer Classification+Logistic Regression')
        print(f'accuracy_score\n{accuracy_score(y_test,y_pred)}')
        print(f'classification report\n{classification_report(y_test,y_pred)}')

        #CountVectorize
        count=self.count_vectorize()
        X_train,X_test,y_train,y_test=train_test_split(count,self.y,test_size=0.3,random_state=42)
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        self.naive_bayes.fit(X_train_res,y_train_res)
        y_pred=self.naive_bayes.predict(X_test)
        print('Countvectorize Classification+Logistic Regression')
        print(f'accuracy_score\n{accuracy_score(y_test,y_pred)}')
        print(f'classification report\n{classification_report(y_test,y_pred)}')
    
        rf=RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(X_train_res, y_train_res)
        y_pred=rf.predict(X_test)
        print('TF-IDF + Random Forest')
        print(f'accuracy_score\n{accuracy_score(y_test, y_pred)}')
        print(f'classification report\n{classification_report(y_test, y_pred)}')
        self.random_forest = rf

    def save_model(self):
        joblib.dump(self.vectorizer,'vectorizer_job_detection.pkl')
        joblib.dump(self.naive_bayes,'naive_bayes_job_detection.pkl')
        joblib.dump(self.sentence_transformer,'sentence_transformer_job_detection.pkl')
        joblib.dump(self.word2vec,'word2vec_job_detection.pkl')
        joblib.dump(self.random_forest, 'random_forest_job_detection.pkl')
        joblib.dump(self.sentence_classifier, 'logistic_regression_sentence.pkl')

if __name__=='__main__':
    t=Train_model()
    t.prepare_data()
    t.train_data()
    t.save_model()
