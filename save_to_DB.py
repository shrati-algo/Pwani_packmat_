import mysql.connector
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Save truck_visit_id, output_path, and object count to the database
def save_video_log(truck_visit_id, output_path, counter):
    try:
        db_host = os.getenv("DB_HOST")
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_name = os.getenv("DB_NAME")

        # Connect to the database
        conn = mysql.connector.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name
        )
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Truck_video_logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                truck_visit_id VARCHAR(255) UNIQUE,
                output_path TEXT,
                object_count INT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            INSERT IGNORE INTO Truck_video_logs (truck_visit_id, output_path, object_count)
            VALUES (%s, %s, %s)
        """, (truck_visit_id, output_path, counter))

        conn.commit()
        print(f"[INFO] Saved truck_visit_id: {truck_visit_id}, path: {output_path}, count: {counter}")

    except mysql.connector.Error as e:
        print(f"[ERROR] MySQL error while saving video log: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()
