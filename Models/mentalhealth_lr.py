!pip install nltk
import nltk 
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from google.colab import files 
upload= files.upload()
data= pd.read_excel('mental health dataset.xlsx')
df=pd.DataFrame(data)

#DELETE NULL VALUES 

null_text_count = df['Text'].isnull().sum()
null_text_count
index = data[data['Text'].isnull()].index
index
df.dropna(axis=0, how='any', inplace=True)
null_text_count = df['Text'].isnull().sum()
null_text_count

#REMOVE DUPLICATES

duplicates = df[df.duplicated(keep=False)]
print("Exact duplicates:")
print(duplicates)
len(df)
df = df.drop_duplicates(keep='first')
len(df)
df['Label'].value_counts()
duplicates = df[df.duplicated(keep=False)]
print("Exact duplicates:")
print(duplicates)

#REMOVE STOP WORDS

