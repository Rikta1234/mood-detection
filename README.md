# === MOOD DETECTION FROM TEXT USING MULTINOMIAL NAIVE BAYES ===

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
data = pd.read_csv("emotions.csv")  # Dataset with 'text' and 'emotion' columns

# Display available emotions
print("Emotions in dataset:", data['emotion'].unique())

# Features and labels
X = data['text']
y = data['emotion']

# Text vectorization using TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_vect = tfidf.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluation
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Test on new text
def predict_mood(text):
    text_transformed = tfidf.transform([text])
    prediction = model.predict(text_transformed)[0]
    return prediction

# Example
sample = "I feel like crying. Nothing is going right."
print(f"\nInput: '{sample}'")
print("Predicted Mood:", predict_mood(sample))
# mood-detection
