from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates folder (your frontend folder)
templates = Jinja2Templates(directory="../frontend")

# Load model
model = joblib.load("model/fake_news_model.pkl")
vectorizer = joblib.load("model/tfidf_vectorizer.pkl")


# -------------------- ROUTES FOR HTML --------------------

@app.get("/", response_class=HTMLResponse)
def read_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/predict", response_class=HTMLResponse)
def read_predict(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
def read_about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/contact", response_class=HTMLResponse)
def read_contact(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})


# -------------------- API PREDICTION --------------------

class NewsInput(BaseModel):
    text: str

@app.post("/api/predict")
def predict_news(data: NewsInput):

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