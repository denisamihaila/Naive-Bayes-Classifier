import json
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')

# Load data from JSON file
with open("articole.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Flatten the data to create a DataFrame
texts = []
labels = []
lemmatizer = WordNetLemmatizer()
for sport, articles in data.items():
    for article in articles:
        # Lowercasing and lemmatizing text
        text = article["text"].lower()
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
        texts.append(text)
        labels.append(sport)

# Create the DataFrame
df = pd.DataFrame({"text": texts, "sport": labels})

# Display prior probabilities for each category
prior_probabilities = df['sport'].value_counts(normalize=True)
print("Prior Probabilities (Priors for each category):\n", prior_probabilities)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['sport'], test_size=0.2, random_state=42)

# Transform the text into TF-IDF vectors with unigrams and bigrams
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Hyperparameter tuning for MultinomialNB
param_grid = {'alpha': [0.1, 0.5, 1.0]}
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5)
grid_search.fit(X_train_vec, y_train)
model = grid_search.best_estimator_
print("Best alpha:", grid_search.best_params_['alpha'])

# Display conditional probabilities for top features in each category
feature_log_prob = model.feature_log_prob_
for i, category in enumerate(model.classes_):
    feature_probs = dict(zip(vectorizer.get_feature_names_out(), feature_log_prob[i]))
    top_features = sorted(feature_probs.items(), key=lambda item: item[1], reverse=True)[:10]
    print(f"\nTop features for '{category}':")
    print(pd.DataFrame(top_features, columns=['Feature', 'Log-Probability']))

# Function to classify a new text with probability output
def classify_text(text):
    # Preprocess the input text (lowercasing and lemmatizing)
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.lower().split()])
    # Transform the text to a vector
    text_vec = vectorizer.transform([text])
    # Predict the category of the text
    prediction = model.predict(text_vec)[0]
    # Get the probabilities for each category
    probabilities = model.predict_proba(text_vec)[0]
    # Convert class labels and probabilities to Python types, rounding probabilities to 3 decimal places
    prob_dict = {str(label): round(float(prob), 3) for label, prob in zip(model.classes_, probabilities)}
    return prediction, prob_dict


# Example of classifying a new text
new_text = "Lionel Messi and his team PSG won a thrilling match in the last minute."
category, prob_dict = classify_text(new_text)
print(f"The new text belongs to the category: {category}")
print("Probabilities for each category:", prob_dict)

# Evaluate the model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
