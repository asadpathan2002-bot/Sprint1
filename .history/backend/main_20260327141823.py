from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from db import get_connection
from pathlib import Path
import joblib
import json
import re
import string
import requests
from difflib import SequenceMatcher
from nltk.corpus import stopwords

# =============================
# Text Cleaning Setup
# =============================
stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# =============================
# News API Setup
# =============================
NEWS_API_KEY = "7dc40be2f09e4fb7a174896665beee35"
NEWS_API_URL = "https://newsapi.org/v2/everything"

def get_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def check_newsapi_for_real_news(user_text):
    try:
        short_query = " ".join(user_text.split()[:12])

        params = {
            "q": short_query,
            "apiKey": NEWS_API_KEY,
            "language": "en",
            "pageSize": 5
        }

        response = requests.get(NEWS_API_URL, params=params, timeout=10)
        data = response.json()

        articles = data.get("articles", [])
        if not articles:
            return False

        for article in articles:
            title = article.get("title", "")
            description = article.get("description", "")
            content = article.get("content", "")

            combined = f"{title} {description} {content}"

            similarity = get_similarity(user_text, combined)

            if similarity >= 0.35:
                return True

        return False

    except Exception as e:
        print("NewsAPI error:", e)
        return False

# =============================
# Test DB Connection
# =============================
try:
    conn = get_connection()
    print("PostgreSQL connected successfully")
    conn.close()
except Exception as e:
    print("DB connection error:", e)

# =============================
# FastAPI App
# =============================
app = FastAPI()

# =============================
# CORS
# =============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# Static & Templates
# =============================
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# =============================
# Load JSON Data
# =============================
with open("data/news.json", "r", encoding="utf-8") as f:
    news_data = json.load(f)

# =============================
# Load Model
# =============================
BASE_DIR = Path(__file__).resolve().parent.parent   # 👈 go one folder up
MODEL_DIR = BASE_DIR / "model"

try:
    model = joblib.load(MODEL_DIR / "fake_news_model.pkl")
    vectorizer = joblib.load(MODEL_DIR / "tfidf_vectorizer.pkl")
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model load error:", e)
    model = None
    vectorizer = None

# =============================
# ROUTES
# =============================
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/news")
async def news_page(request: Request, search: str = None, category: str = None):
    filtered = news_data

    if search:
        filtered = [n for n in filtered if search.lower() in n["title"].lower()]

    if category:
        filtered = [n for n in filtered if n["category"].lower() == category.lower()]

    return templates.TemplateResponse("news.html", {"request": request, "news": filtered})

@app.get("/article/{id}")
async def article_page(request: Request, id: int):
    article = next((n for n in news_data if n["id"] == id), None)
    return templates.TemplateResponse("article.html", {"request": request, "article": article})

@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/contact", response_class=HTMLResponse)
async def contact_page(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})

# =============================
# API
# =============================
class NewsInput(BaseModel):
    text: str

@app.post("/api/predict")
def predict_news(data: NewsInput):

    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not data.text.strip():
        raise HTTPException(status_code=400, detail="Empty input")

    # 🔥 STEP 1: NewsAPI check
    if check_newsapi_for_real_news(data.text):
        result = "Real News"
        confidence = 95.0

    else:
        # 🔥 STEP 2: ML model
        cleaned = clean_text(data.text)
        vec = vectorizer.transform([cleaned])

        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0].max()

        result = "Real News" if pred == 1 else "Fake News"
        confidence = round(float(prob) * 100, 2)

    # =============================
    # Save to DB
    # =============================
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO prediction_history (article_text, prediction, confidence) VALUES (%s, %s, %s)",
            (data.text, result, confidence)
        )

        conn.commit()
        cursor.close()
        conn.close()

    except Exception as e:
        print("DB error:", e)

    return {
        "prediction": result,
        "confidence": confidence
    }