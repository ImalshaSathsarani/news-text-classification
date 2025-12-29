import pandas as pd 
import numpy as np
import re
import joblib
import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

from preprocessing import load_and_preprocess_data
from evaluate import evaluate_models_with_cv

#Load and preprocess
df, target_names = load_and_preprocess_data()
print(f"Data Loaded:{df.shape}")

#Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['target'], test_size=0.2, random_state=42, stratify=df['target'])

#Define pipeline for 3 models
pipelines = {
    'Naive_Bayes':Pipeline([('tfidf', TfidfVectorizer(stop_words='english')), ('clf',MultinomialNB())]),
    'Logistic_Regression':Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),('clf', LogisticRegression(max_iter=1000))]),
    'SVM':Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),('clf', LinearSVC(dual='auto'))])
}

#Hyperparameter grids
param_grids = {
    'Naive_Bayes':{
        'tfidf__ngram_range':[(1,1),(1,2)],
        'tfidf__max_df':[0.9,1.0],
        'clf__alpha':[0.1,0.5,1.0]
    },
    'Logistic_Regression':{
        'tfidf__ngram_range':[(1,1),(1,2)],
        'tfidf__max_df':[0.9,1.0],
        'clf__C':[0.1,1.0,10.0]
    },
    'SVM':{
        'tfidf__ngram_range':[(1,1),(1,2)],
        'tfidf__max_df':[0.9,1.0],
        'clf__C':[0.1,1.0,10.0]
    }
}

#Evaluate with CV
best_model, best_model_name, _ = evaluate_models_with_cv(
    pipelines,
    param_grids,
    X_train,
    X_test,
    y_train,
    y_test,
    target_names
)

save_path = '../models/best_text_classification_model.pkl'
os.makedirs(os.path.dirname(save_path), exist_ok=True)

#Save the best model
joblib.dump(best_model, save_path)
print(f"\nBest model '{best_model_name}' saved as 'best_text_classification_model.pkl' in the {save_path} directory.")