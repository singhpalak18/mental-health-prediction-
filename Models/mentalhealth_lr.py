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

def remove_stopwords(Text):
    words = nltk.word_tokenize(Text)
    words = [word for word in words if word.lower() not in stopwords.words("english")]
    return " ".join(words)

df['Text'] = df['Text'].apply(remove_stopwords)
df.to_csv("mentaldataset_cleaned.csv", index=False)
df.to_csv("mentaldataset_cleaned.csv",index=False)
df.to_excel("mentaldataset_cleaned.xlsx",index=False)
data=pd.read_csv("mentaldataset_cleaned.csv")
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X = tfidf_vectorizer.fit_transform(data['Text'])
y = data['Label']

#REGRESSION 

tfidf_vectorizer = TfidfVectorizer(max_features=1000)
X = tfidf_vectorizer.fit_transform(data['Text'])
y = data['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier= LogisticRegression()
classifier.fit(X, y)
questions = [
    "Do you ever worry about the security of your job?",
"Do you feel confident at work? ",
"Have you started to avoid spending time with friends and loved ones due to extensive work?",
"Do you feel like you have a good work life balance here?",
"Do you become irritable or annoyed more quickly than I have in the past?",
"Do you often feel restless, on edge, or unable to relax?",
"Do you feel comfortable talking about my mental health with others inside our organisation?",
"Is it difficult to fall asleep, get enough sleep, or wake up on time most days?",
"Do you ever feel overworked or underworked here as an employee?",
"Do you feel that your work is not recognized or underappreciated?"
]
user_responses = []
predicted_labels = []

for i, question in enumerate(questions):
    response = input(f"Q{i + 1}: {question} ")
    user_responses.append(response)
    response_tfidf = tfidf_vectorizer.transform([response])
    predicted_label = classifier.predict(response_tfidf)[0]
    predicted_labels.append(predicted_label)
print("Predicted Labels for Each Question:")
for i, label in enumerate(predicted_labels):
    print(f"Q{i + 1}: {label}")

y_pred = classifier.predict(X_test)
classification_rep = classification_report(y_test, y_pred)

print("\nClassification Report:")
print(classification_rep)

#DISPLAY STATS USING PIE CHART 

label_counts = pd.Series(predicted_labels).value_counts()
labels = label_counts.index
sizes = label_counts.values
colors=['purple', 'pink', 'cyan','red','blue','brown']
plt.pie(sizes, labels=labels, colors=colors, startangle=90, shadow=True,explode=(0.1,)*len(labels), autopct='%1.2f%%')
plt.axis('equal')

plt.title('Composition of Labels for All Questions')
plt.show()
