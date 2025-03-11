import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load training data
with open("data/intents.json", "r") as file:
    data = json.load(file)

# Preprocess data
tags = []
patterns = []
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tags.append(intent["tag"])
        patterns.append(pattern)

# Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)

# Train Logistic Regression classifier
model = LogisticRegression()
model.fit(X, tags)

# Save the model
with open("model.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)

print("✅ Training completed! Model saved.")
