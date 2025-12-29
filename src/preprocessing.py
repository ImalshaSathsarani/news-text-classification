
import pandas as pd
import re
from sklearn.datasets import fetch_20newsgroups


#Load data (We use only 4 categories to keep it fast and clean)
categories = ['sci.space','comp.graphics','talk.politics.mideast','rec.sport.hockey']
newsgroups = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes')) 

df = pd.DataFrame({'text': newsgroups.data, 'target': newsgroups.target})
print(f"Data Loaded: {df.shape}")

def clean_text(text):
    #1.Lowercase
    text = text.lower()
    #Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    #Remove multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_and_preprocess_data():
    newsgroups = fetch_20newsgroups(
        subset='all',
        categories=categories,
        remove=('headers', 'footers', 'quotes')
    )

    df = pd.DataFrame({
        'text':newsgroups.data,
        'target':newsgroups.target
    })

    df['clean_text'] = df['text'].apply(clean_text)

    return df, newsgroups.target_names