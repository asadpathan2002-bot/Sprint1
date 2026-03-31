from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib

# -----------------------------
# Create FastAPI App
# -----------------------------
app = FastAPI()

# -----------------------------
# Enable CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Mount Static Folder
# -----------------------------
app.mount("/static", StaticFiles(directory="static"), name="static")

# -----------------------------
# Templates Folder
# -----------------------------
templates = Jinja2Templates(directory="templates")

# -----------------------------
# Load Model & Vectorizer
# -----------------------------
try:
    model = joblib.load("model/fake_news_model.pkl")
    vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
except Exception as e:
    print("Error loading model:", e)
    model = None
    vectorizer = None

# -----------------------------
# Page Routes
# -----------------------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predict", response_class=HTMLResponse)
def predict_page(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
def about_page(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/contact", response_class=HTMLResponse)
def contact_page(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})

# -----------------------------
# API Root Check
# -----------------------------
@app.get("/api")
def api_home():
    return {"message": "TruthLens Fake News Detection API is running 🚀"}

# -----------------------------
# Request Body Model
# -----------------------------
class NewsInput(BaseModel):
    text: str

# -----------------------------
# Prediction Endpoint
# -----------------------------
@app.post("/api/predict")
def predict_news(data: NewsInput):

    if model is None or vectorizer is None:
        raise HTTPException(status_code=500, detail="Model not loaded properly")

    if not data.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    vectorized_text = vectorizer.transform([data.text])

    prediction = model.predict(vectorized_text)[0]
    probability = model.predict_proba(vectorized_text)[0].max()

    result = "Real News" if prediction == 1 else "Fake News"

    return {
        "prediction": result,
        "confidence": round(float(probability) * 100, 2)
    }