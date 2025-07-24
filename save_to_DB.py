import mysql.connector

# Save truck_visit_id, output_path, and object count to the database
def save_video_log(truck_visit_id, output_path, counter):
    try:
        conn = mysql.connector.connect(
            host="192.168.5.82",
            user="root",
            password="N47309HxFWE2Ehc",
            database="pwani_scm"
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
