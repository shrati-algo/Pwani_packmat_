import os
import time
from datetime import datetime, timedelta

def delete_old_full_videos(folder_path, days=3):
    """
    Deletes files ending with '_full.mp4' older than given number of days.
    """

    if not os.path.exists(folder_path):
        print(f"Folder does not exist: {folder_path}")
        return

    # Time threshold (3 days ago)
    cutoff_time = time.time() - (days * 24 * 60 * 60)

    deleted_count = 0

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check: file, ends with _full.mp4
        if os.path.isfile(file_path) and filename.endswith("_full.mp4"):
            
            # Get file modification time
            file_mtime = os.path.getmtime(file_path)

            if file_mtime < cutoff_time:
                try:
                    os.remove(file_path)
                    print(f"Deleted: {filename}")
                    deleted_count += 1
                except Exception as e:
                    print(f"Error deleting {filename}: {e}")

    print(f"\nTotal deleted files: {deleted_count}")


# # 🔹 CHANGE THIS to your folder path
# folder_to_clean = r"outputs"

# delete_old_full_videos(folder_to_clean, days=3)
