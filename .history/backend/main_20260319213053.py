from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from db import get_connection
import joblib
import json
import re
import string
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
# Test PostgreSQL Connection
# =============================
try:
    conn = get_connection()
    print("PostgreSQL connected successfully")
    conn.close()
except Exception as e:
    print("PostgreSQL connection error:", e)

# =============================
# Create FastAPI App
# =============================
app = FastAPI()

# =============================
# Enable CORS
# =============================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================
# Static Folder
# =============================
app.mount("/static", StaticFiles(directory="static"), name="static")

# =============================
# Templates Folder
# =============================
templates = Jinja2Templates(directory="templates")

# =============================
# Load JSON Data
# =============================
with open("data/news.json", "r", encoding="utf-8") as f:
    news_data = json.load(f)

# =============================
# Load Model & Vectorizer
# =============================
try:
    model = joblib.load("../model/fake_news_model.pkl")
    vectorizer = joblib.load("../model/tfidf_vectorizer.pkl")
except Exception as e:
    print("Error loading model:", e)
    model = None
    vectorizer = None

# =====================================================
#               PAGE ROUTES
# =====================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/news")
async def news_page(request: Request, search: str = None, category: str = None):
    filtered_news = news_data

    if search:
        filtered_news = [
            n for n in filtered_news
            if search.lower() in n["title"].lower()
        ]

    if category:
        filtered_news = [
            n for n in filtered_news
            if n["category"].lower() == category.lower()
        ]

    return templates.TemplateResponse(
        "news.html",
        {
            "request": request,
            "news": filtered_news
        }
    )

@app.get("/article/{id}")
async def article_page(request: Request, id: int):
    article = None

    for n in news_data:
        if n["id"] == id:
            article = n
            break

    return templates.TemplateResponse(
        "article.html",
        {
            "request": request,
            "article": article
        }
    )

@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/contact", response_class=HTMLResponse)
async def contact_page(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})

# =====================================================
#               API ROUTE (Prediction)
# =====================================================

class NewsInput(BaseModel):
    text: str

@app.post("/api/predict")
def predict_news(data: NewsInput):

    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly")

    if not data.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    cleaned = clean_text(data.text)
    vectorized_text = vectorizer.transform([cleaned])

    prediction = model.predict(vectorized_text)[0]
    probability = model.predict_proba(vectorized_text)[0].max()

    result = "Real News" if prediction == 1 else "Fake News"
    confidence = round(float(probability) * 100, 2)

    # Save prediction to PostgreSQL
    try:
        print("🔥 Saving to DB...")

        conn = get_connection()
        cursor = conn.cursor()

        sql = """
            INSERT INTO prediction_history (article_text, prediction, confidence)
            VALUES (%s, %s, %s)
        """
        values = (data.text, result, confidence)

        cursor.execute(sql, values)
        conn.commit()

        print("✅ Data saved")

        cursor.close()
        conn.close()

    except Exception as e:
        print("❌ DB ERROR:", e)

    return {
        "prediction": result,
        "confidence": confidence
    }