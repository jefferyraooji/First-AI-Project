# 🤖 First-AI-Project (ML-Based Chatbot)

A **machine learning-based** NLP chatbot that uses **TF-IDF + Logistic Regression** for **intent classification**.

## 🚀 Features
- Detects user intent (greetings, farewells, weather inquiry, etc.)
- Responds appropriately based on trained patterns
- Lightweight and easily extendable

## 📂 Project Structure
- **First-AI-Project/**
  - 📂 `data/`
    - 📄 `intents.json` - Training data
  - 📄 `chatbot.py` - Chatbot script
  - 📄 `train.py` - Model training
  - 📄 `model.pkl` - Trained model
  - 📄 `README.md` - Project documentation
  - 📄 `requirements.txt` - Dependencies

## 🔧 Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_GITHUB_USERNAME/First-AI-Project.git
cd First-AI-Project

# Install dependencies
pip install -r requirements.txt
```
## 🏈 Train the Model
```bash
python train.py
```
## 💬 Run the Chatbot
```bash
python chatbot.py
```
## 📉 Dependencies
- scikit-learn
- numpy
- json
- pickle
## 📚 Future Improvements
- Integrate OpenAI GPT-4 for smarter responses
- Use LSTM/RNN for better intent classification
- Connect to an API for real-time weather updates

## 🌟 Example Interaction
```bash
You: Hello
🤖 Chatbot: Hi there!

You: What's the weather like today?
🤖 Chatbot: I can't check the weather right now, but you can visit a weather website!

You: quit
🤖 Chatbot: Goodbye!
```
Let me know if you want to give me some advices! 🚀🔥
