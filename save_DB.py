import mysql.connector
from dotenv import load_dotenv
import os

load_dotenv()

def _get_conn():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
    )

def save_video_log_start(truck_visit_id, truck_product_visit_id):
    """
    Insert row at START with object_count=0.
    Returns inserted row id.
    """
    conn = None
    cursor = None
    try:
        conn = _get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO Truck_video_logs
            (truck_visit_id, truck_product_visit_id, output_path, object_count, video_link, createdAt, updatedAt)
            VALUES (%s, %s, %s, %s, %s, NOW(), NOW())
        """, (truck_visit_id, truck_product_visit_id, None, 0, None))

        conn.commit()
        return cursor.lastrowid

    except mysql.connector.Error as e:
        print(f"[ERROR] MySQL error while START logging: {e}")
        return None

    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()


def update_video_log_end(log_id, output_path, object_count, video_link, end_pressed_at_iso):
    """
    Update the same row on END, and set updatedAt to END pressed timestamp.
    """
    conn = None
    cursor = None
    try:
        conn = _get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE Truck_video_logs
            SET output_path=%s,
                object_count=%s,
                video_link=%s,
                updatedAt=%s
            WHERE id=%s
        """, (output_path, int(object_count or 0), video_link, end_pressed_at_iso, int(log_id)))

        conn.commit()
        return True

    except mysql.connector.Error as e:
        print(f"[ERROR] MySQL error while END updating: {e}")
        return False

    finally:
        if conn and conn.is_connected():
            cursor.close()
            conn.close()
