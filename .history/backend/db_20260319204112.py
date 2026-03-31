import psycopg2

def get_connection():
    conn = psycopg2.connect(
        host="localhost",
        database="fake_news_db",
        user="postgres",
        password="sanika123",   # change this
        port="5432"
    )
    return conn