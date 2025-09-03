import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify

# .env dosyasını yükle
load_dotenv()

# 1. API Anahtarını .env dosyasından al
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

# CSV dosyasını yükle
try:
    df = pd.read_csv('questions.csv')
    questions = df['question'].tolist()
    answers = df['answer'].tolist()
    print("CSV dosyası başarıyla yüklendi.")
except FileNotFoundError:
    print("Hata: 'questions.csv' dosyası bulunamadı. Lütfen dosyanın doğru yolda olduğundan emin olun.")
    exit()

# TF-IDF vektörleştiriciyi başlat
vectorizer = TfidfVectorizer().fit(questions)
question_vectors = vectorizer.transform(questions)

def get_gemini_response(prompt):
    """
    Gemini API'ye bağlanarak yanıt alır.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Üzgünüm, Gemini'ye bağlanırken bir hata oluştu: {e}"

# Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]

    if "karaciğer" in user_input.lower():
        user_vector = vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vector, question_vectors)
        best_match_index = np.argmax(similarities)
        best_match_score = similarities[0][best_match_index]

        if best_match_score > 0.5:
            answer = answers[best_match_index]
        else:
            answer = get_gemini_response(user_input)
    else:
        answer = get_gemini_response(user_input)

    return jsonify({"answer": answer})

if __name__ == "__main__":
    if not API_KEY:
        print("Hata: 'GEMINI_API_KEY' .env dosyasında bulunamadı.")
    else:
        app.run(debug=True)   