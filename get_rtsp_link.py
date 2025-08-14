import mysql.connector
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

def get_rtsp_link(camera_id):
    try:
        # Fetch credentials from environment variables
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

        # Fetch RTSP link for the given camera ID
        query = "SELECT rtspLink FROM OffloadingRtsp WHERE id = %s"
        cursor.execute(query, (camera_id,))
        result = cursor.fetchone()

        cursor.close()
        conn.close()

        if result:
            return result[0]
        else:
            raise ValueError(f"No RTSP link found for camera ID {camera_id}.")

    except mysql.connector.Error as db_err:
        raise ConnectionError(f"Database error: {db_err}")
    except Exception as e:
        raise e


# if __name__ == "__main__":
#     try:
#         camera_id = int(input("Enter camera ID: "))
#         rtsp_link = get_rtsp_link(camera_id)
#         print(f"RTSP Link for camera ID {camera_id}: {rtsp_link}")

#     except ValueError as ve:
#         print(f"[ERROR] {ve}")
#     except ConnectionError as ce:
#         print(f"[ERROR] {ce}")
#     except Exception as e:
#         print(f"[ERROR] Unexpected error: {e}")

# print(get_rtsp_link(4))