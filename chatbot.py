import pickle
import json
import random

# Load trained model
with open("model.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

# Load intents data
with open("data/intents.json", "r") as file:
    data = json.load(file)

# Function to find response based on tag
def get_response(tag):
    for intent in data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I'm not sure how to respond to that."

# Chatbot function
def chat():
    print("🤖 Chatbot: Hello! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("🤖 Chatbot: Goodbye!")
            break
        X_input = vectorizer.transform([user_input])
        tag = model.predict(X_input)[0]
        response = get_response(tag)
        print(f"🤖 Chatbot: {response}")

if __name__ == "__main__":
    chat()
