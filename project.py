import pandas as pd
import numpy as np


import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score , classification_report

df= pd.read_csv(r"C:\Users\abdel\Spam detection\deploy\spam.csv",encoding="latin-1")
df=df.drop(["Unnamed: 2" , "Unnamed: 3" , "Unnamed: 4"] , axis=1)
df=df.rename(columns={"v1": "Target" , "v2": "Text"})

def text_cleaning(text):
    
    #Converting text into lowercase
    text = str(text).lower()
    
    #Removing square brackets from the text
    text = re.sub('\[.*?\]','',text)
    
    
    #Removing links starting with (https or www)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    
    #Removing <"text"> type of text 
    text = re.sub('<.*?>+','',text)
    
    #Removing punctuations
    text = re.sub("[%s]" % re.escape(string.punctuation),'',text)
    
    #Removing new lines
    text = re.sub("\n",'',text)
    
    #Removing alphanumeric numbers 
    text = re.sub('\w*\d\w*','',text)
    
    return(text)

df['cleaned_text']=df['Text'].apply(text_cleaning)

wordnet = WordNetLemmatizer()
def remove_stopwords(text):
    text = text.split()
    text = [wordnet.lemmatize(word) for word in text if word not in set(stopwords.words('english'))]
    text = " ".join(text)
    return(text)

df['cleaned_text']=df['cleaned_text'].apply(remove_stopwords)

le=LabelEncoder()
df['Target']=le.fit_transform(df["Target"])


x=df['cleaned_text']
y=df['Target']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)

cv=CountVectorizer()
vect=cv.fit(x_train)
x_train_vector=vect.transform(x_train)
x_test_vector=vect.transform(x_test)

def model_spam (text):
    clean_done=[]
    entry=[text]
    clean=text_cleaning(entry[0])
    clean_done.append(clean)
    sample=vect.transform(clean_done)
    naive=MultinomialNB()
    model=naive.fit(x_train_vector,y_train)
    return model.predict(sample)

