import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
import math

# 1. Load and preprocess the data
with open("articole.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = []
labels = []
for sport, articles in data.items():
    for article in articles:
        texts.append(article["text"].lower())  # Lowercase text
        labels.append(sport)

# Create DataFrame
df = pd.DataFrame({"text": texts, "sport": labels})

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sport'], test_size=0.2, random_state=42)

# 2. Vectorize the text using CountVectorizer to get word counts
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
vocab = vectorizer.get_feature_names_out()

# Convert to array for easier access
X_train_array = X_train_vec.toarray()
X_test_array = X_test_vec.toarray()

# 3. Implementing Naive Bayes manually
class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Laplace smoothing parameter
        self.priors = {}
        self.word_counts = {}
        self.total_words = {}
        self.vocab_size = 0
        self.classes = []
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.vocab_size = X.shape[1]  # Number of unique words in the vocabulary
        
        # Initialize dictionaries to hold word counts and total word counts per class
        self.word_counts = {category: np.zeros(self.vocab_size) for category in self.classes}
        self.total_words = {category: 0 for category in self.classes}
        self.priors = {category: 0 for category in self.classes}

        # Calculate priors and word counts for each class
        for category in self.classes:
            X_category = X[y == category]
            self.priors[category] = X_category.shape[0] / X.shape[0]  # P(category)
            self.word_counts[category] = X_category.sum(axis=0)  # Sum of word counts per word in the category
            self.total_words[category] = self.word_counts[category].sum()  # Total word count in the category
            
    def predict(self, X):
        predictions = []
        for x in X:
            class_probs = {}
            for category in self.classes:
                # Start with the log of the prior probability
                log_prob = math.log(self.priors[category])
                
                # Add log probability of each word in the document given the category
                for i in range(self.vocab_size):
                    word_count = x[i]
                    # Calculate P(word | category) with Laplace smoothing
                    word_prob = (self.word_counts[category][i] + self.alpha) / \
                                (self.total_words[category] + self.alpha * self.vocab_size)
                    log_prob += word_count * math.log(word_prob)  # Use log for numerical stability
                    
                class_probs[category] = log_prob
            # Choose the class with the highest probability
            predictions.append(max(class_probs, key=class_probs.get))
        return predictions
    
    def predict_proba(self, X):
        probabilities = []
        for x in X:
            class_probs = {}
            for category in self.classes:
                # Start with the log of the prior probability
                log_prob = math.log(self.priors[category])
                
                # Add log probability of each word in the document given the category
                for i in range(self.vocab_size):
                    word_count = x[i]
                    word_prob = (self.word_counts[category][i] + self.alpha) / \
                                (self.total_words[category] + self.alpha * self.vocab_size)
                    log_prob += word_count * math.log(word_prob)
                    
                # Convert log probabilities to probabilities
                class_probs[category] = math.exp(log_prob)
                
            # Normalize probabilities
            total_prob = sum(class_probs.values())
            class_probs = {category: prob / total_prob for category, prob in class_probs.items()}
            probabilities.append(class_probs)
        return probabilities

# 4. Train the custom Naive Bayes model
nb_model = MultinomialNaiveBayes(alpha=1.0)
nb_model.fit(X_train_array, y_train)

# 5. Evaluate the model
y_pred = nb_model.predict(X_test_array)

# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Confusion matrix and classification report
from sklearn.metrics import classification_report, confusion_matrix
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 6. Predict and show probabilities for a new text
def classify_text(text):
    text_vec = vectorizer.transform([text.lower()]).toarray()
    prediction = nb_model.predict(text_vec)[0]
    probabilities = nb_model.predict_proba(text_vec)[0]
    return prediction, probabilities

# Example classification
new_text = "Lionel Messi and his team PSG won a thrilling match in the last minute."
category, prob_dict = classify_text(new_text)
print(f"The new text belongs to the category: {category}")
print("Probabilities for each category:", prob_dict)
