import os
import sqlite3
from datetime import datetime

# Setup DB path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_dir = os.path.join(base_dir, 'database')
db_path = os.path.join(db_dir, 'logs.db')

def get_db_connection():
    """Returns a connection to the SQLite database."""
    os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database schemas for predictions logging and user feedbacks."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Prediction Logs Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prediction_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        review_text TEXT NOT NULL,
        cleaned_text TEXT,
        predicted_sentiment TEXT NOT NULL,
        confidence REAL NOT NULL,
        model_name TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """)
    
    # 2. Feedback Logs Table (For user corrections/audits)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedback_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        log_id INTEGER NOT NULL,
        correct_sentiment TEXT NOT NULL,
        notes TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY (log_id) REFERENCES prediction_logs (id)
    )
    """)
    
    conn.commit()
    conn.close()
    print(f"[DATABASE] SQLite database initialized at {db_path}")

def log_prediction(review_text, cleaned_text, predicted_sentiment, confidence, model_name):
    """Inserts a transaction record of a sentiment prediction."""
    conn = get_db_connection()
    cursor = conn.cursor()
    created_at = datetime.now().isoformat()
    
    cursor.execute("""
    INSERT INTO prediction_logs (review_text, cleaned_text, predicted_sentiment, confidence, model_name, created_at)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (review_text, cleaned_text, predicted_sentiment, confidence, model_name, created_at))
    
    last_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return last_id

def log_feedback(log_id, correct_sentiment, notes=None):
    """Logs correction feedback for auditing model performance."""
    conn = get_db_connection()
    cursor = conn.cursor()
    created_at = datetime.now().isoformat()
    
    cursor.execute("""
    INSERT INTO feedback_logs (log_id, correct_sentiment, notes, created_at)
    VALUES (?, ?, ?, ?)
    """, (log_id, correct_sentiment, notes, created_at))
    
    conn.commit()
    conn.close()

def get_stats():
    """Retrieves operational metrics for corporate reporting."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Total predictions
    cursor.execute("SELECT COUNT(*) FROM prediction_logs")
    total_predictions = cursor.fetchone()[0]
    
    # Distribution of predicted sentiments
    cursor.execute("SELECT predicted_sentiment, COUNT(*) FROM prediction_logs GROUP BY predicted_sentiment")
    sentiment_dist = {row[0]: row[1] for row in cursor.fetchall()}
    
    # Total feedbacks received
    cursor.execute("SELECT COUNT(*) FROM feedback_logs")
    total_feedbacks = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "total_predictions": total_predictions,
        "sentiment_distribution": {
            "Positif": sentiment_dist.get("Positif", 0),
            "Netral": sentiment_dist.get("Netral", 0),
            "Negatif": sentiment_dist.get("Negatif", 0)
        },
        "total_feedbacks": total_feedbacks
    }

def get_recent_logs(limit=25):
    """Retrieves the recent transaction logs for visualization in the dashboard table."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
    SELECT p.id, p.review_text, p.predicted_sentiment, p.confidence, p.created_at, f.correct_sentiment
    FROM prediction_logs p
    LEFT JOIN feedback_logs f ON p.id = f.log_id
    ORDER BY p.id DESC
    LIMIT ?
    """, (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]
