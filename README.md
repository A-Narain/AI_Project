HATE SPEECH DETECTION SYSTEM

PROJECT OVERVIEW

A machine learning based web application that classifies input text as:

Hate Speech
Offensive Language
Neutral Content

The system uses natural language processing techniques and a trained classification model, exposed through a backend API and a frontend interface.

PROBLEM STATEMENT

Online platforms generate large volumes of user content, making manual moderation inefficient. This project automates detection of harmful text to support safer digital environments.

SOLUTION SUMMARY

The system processes user input text, applies NLP preprocessing, converts text into numerical features, and uses a trained machine learning model to predict the category. Results are displayed through a web interface.

TECH STACK

Backend and Machine Learning:
Python
Scikit-learn or TensorFlow or PyTorch (depending on implementation)
Natural Language Processing (NLP)
Flask or FastAPI
TF-IDF or Embeddings

Frontend:
React.js
JavaScript
HTML
CSS

Tools:
Pandas
NumPy
Git
GitHub

SYSTEM ARCHITECTURE
User enters text in the frontend interface
Input is sent to backend via REST API
Text is preprocessed and cleaned
Features are extracted using TF-IDF or embeddings
Machine learning model predicts output class
Result is returned and displayed on UI
PROJECT STRUCTURE

hate-speech-detection

backend
model
app.py
requirements.txt

frontend
src
package.json

README file

SETUP INSTRUCTIONS

BACKEND SETUP

cd backend
pip install -r requirements.txt
python app.py

Backend runs at:
http://127.0.0.1:5000

FRONTEND SETUP

cd frontend
npm install
npm start

Frontend runs at:
http://localhost:3000

KEY FEATURES

Real time text classification
Machine learning based NLP pipeline
REST API integration between frontend and backend
Modular and scalable architecture
Simple user interface

LIMITATIONS

Model performance depends on training dataset quality
Difficulty handling sarcasm and contextual meaning
Requires further optimization for production use

FUTURE IMPROVEMENTS

Integration of SHAP or LIME for explainability
Upgrade to transformer models such as BERT or RoBERTa
Cloud deployment using AWS or Render or Vercel
Analytics dashboard for predictions
Multilingual support

AUTHOR

Aastha Narain
GitHub: https://github.com/A-Narain
