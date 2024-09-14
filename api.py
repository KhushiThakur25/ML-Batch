import streamlit as st
import pandas as pds
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer and stop words
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Load data and preprocess
def load_and_preprocess_data(data):
    data['text'] = data['text'].apply(preprocess_text)
    return data

# Train model
def train_model(df):
    X = df['text']
    y = df['label_col']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    pac = PassiveAggressiveClassifier(max_iter=50)
    pac.fit(X_train_tfidf, y_train)
    
    y_pred = pac.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    
    return tfidf_vectorizer, pac, accuracy, confusion_matrix(y_test, y_pred)

# Prediction function
def predict_fake_news(text, tfidf_vectorizer, model):
    preprocessed_text = preprocess_text(text)
    text_tfidf = tfidf_vectorizer.transform([preprocessed_text])
    prediction = model.predict(text_tfidf)[0]
    return "Potentially Fake" if prediction == 0 else "Likely Reliable"

# Streamlit UI
st.title('Fake News Detection App')

# Load Data

data = pds.read_csv('fake.csv').replace('null', np.nan)

# data = pd.read_csv(uploaded_file)
label = np.random.randint(0, 2, len(data))  # Simulating labels
data['label_col'] = label

# Preprocess and train
df = load_and_preprocess_data(data)
tfidf_vectorizer, model, accuracy, conf_matrix = train_model(df)

st.write(f"Model Accuracy: {accuracy:.2f}")
st.write("Confusion Matrix:")
st.write(conf_matrix)

# Input from user
user_input = st.text_area("Enter news text to check:", "")
if st.button("Predict"):
    if user_input:
        result = predict_fake_news(user_input, tfidf_vectorizer, model)
        st.write(f"Prediction: {result}")
    else:
        st.write("Please enter text to check.")

