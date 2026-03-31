# 📰 TruthLens – Fake News Detection System

## 📌 Project Description

TruthLens is a web-based Fake News Detection System that helps users identify whether a news article is real or fake using Machine Learning.

With the rapid spread of misinformation on social media, it has become difficult for users to verify the authenticity of news. This project aims to solve this problem by providing an easy-to-use platform where users can input news content and get instant predictions.

The system uses Natural Language Processing (NLP) and a trained Logistic Regression model to analyze the text and classify it as Real or Fake along with a confidence score.

---

## 🚀 Features

- User can enter news article through website  
- Frontend sends request to backend via API  
- Backend processes input and predicts result  
- Displays Real/Fake prediction with confidence score  
- Stores prediction history in PostgreSQL  
- Shows Viral Fake News section (data stored in JSON)  

---

## 🛠️ Tech Stack

- Programming Language: Python  
- Frontend: HTML, CSS  
- Backend: FastAPI  
- Database: PostgreSQL  
- Machine Learning: scikit-learn (Logistic Regression)  
- Libraries: Pandas, NumPy, Joblib  
- Vectorization: TF-IDF  
- Server: Uvicorn  

---

## 🤖 Machine Learning Details

- Model Used: Logistic Regression  
- Text Vectorization: TF-IDF  
- Data Preprocessing:
  - Lowercasing  
  - Removing punctuation  
  - Removing numbers  
  - Removing stopwords  
- Train-Test Split: 80:20  

---

## 📂 Dataset

- Source: Kaggle  
- Files Used:
  - Fake.csv  
  - True.csv  
- Processed File:
  - cleaned_data.csv  

---

## 🔄 Project Workflow

1. Collect dataset from Kaggle  
2. Perform data cleaning & preprocessing  
3. Split data into training and testing  
4. Apply TF-IDF vectorization  
5. Train Logistic Regression model  
6. Save model using Joblib  
7. User enters news article via website  
8. Backend processes input and predicts result  
9. Display prediction (Real/Fake + confidence)  
10. Store prediction history in PostgreSQL  
11. Show viral fake news from JSON  

---

## 🗄️ Database

### PostgreSQL (Prediction History)

Stores:
- id  
- news_text  
- prediction  
- confidence  
- created_at  

### JSON

Used for:
- Viral Fake News Section  

---

## ▶️ How to Run Project Locally

Step 1: Install dependencies  
pip install -r requirements.txt  

Step 2: Train model (if needed)  
cd backend  
python model/train.py  

Step 3: Run backend server  
uvicorn main:app --reload  

Step 4: Open in browser  
http://127.0.0.1:8000  

---

## 📁 Project Structure

backend/  
│── main.py  
│── db.py  
│── data/  
│   └── news.json  
│── static/  
│── templates/  

dataset/  
│── Fake.csv  
│── True.csv  
│── cleaned_data.csv  

model/  
│── fake_news_model.pkl  
│── tfidf_vectorizer.pkl  
│── preprocessing.py  
│── train_model.py  

requirements.txt  
README.md  

---

## 🔮 Future Scope

- Improve model accuracy  
- Add Admin Panel  
- Support multiple languages  
- Integrate real-time news APIs  

---

## 👨‍💻 Author

Developed by Sanika Patil