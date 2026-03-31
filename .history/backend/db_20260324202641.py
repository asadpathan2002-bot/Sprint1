import psycopg2

def get_connection():
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="fake_news_db",
            user="postgres",
            password="sanika123",
            port="5432"
        )
        print("✅ Connected successfully")
        return conn

    except Exception as e:
        print("❌ Connection failed:", e)
        return None


# test run
conn = get_connection()