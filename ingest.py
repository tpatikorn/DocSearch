import os
import threading

from db import execute_query, error_log, debug_print
from drive_utils import get_drive_service, list_files_in_folder, download_file
from processor import get_or_create_folder, process_document, get_sharded_path

# Global state for ingestion tracking
ingestion_lock = threading.Lock()
ingestion_status = {
    "is_running": False,
    "current_file": "",
    "processed_count": 0,
    "folder_id": None
}


def run_ingestion_thread(folder_link, mode="Fast"):
    global ingestion_status

    if not ingestion_lock.acquire(blocking=False):
        return

    try:
        ingestion_status["is_running"] = True
        ingestion_status["processed_count"] = 0
        ingestion_status["current_file"] = "Traversing..."

        # Extract folder ID from link
        folder_id = folder_link.split('/')[-1].split('?')[0]
        ingestion_status["folder_id"] = folder_id

        service = get_drive_service()
        processed_drive_ids = []

        # Fetch top-level folder name
        try:
            top_folder = service.files().get(fileId=folder_id, fields='name').execute()
            root_name = top_folder.get('name', 'Root')
        except Exception as e:
            error_log("Error fetching root folder name", str(e))
            root_name = "Root"

        # Initialize root folder in DB
        root_db_id = get_or_create_folder(folder_id, root_name, None)

        def process_recursive(current_folder_id, current_db_id, path_tags):
            try:
                items = list_files_in_folder(service, current_folder_id)
                for item in items:
                    drive_id = item['id']
                    filename = item['name']

                    if item['mimeType'] == 'application/vnd.google-apps.folder':
                        debug_print(f"Entering folder: {filename}")
                        # Create/get subfolder in DB
                        subfolder_db_id = get_or_create_folder(drive_id, filename, current_db_id)
                        # Add current filename to path_tags
                        process_recursive(drive_id, subfolder_db_id, path_tags + [filename])
                    else:
                        processed_drive_ids.append(drive_id)

                        # Check DB
                        existing = execute_query("SELECT id FROM documents WHERE drive_id = %s", (drive_id,))

                        should_process = True
                        if mode == "Fast" and existing:
                            should_process = False
                            debug_print(f"Skipping {filename} (Fast Mode)")

                        if should_process:
                            debug_print(f"Processing {filename}...")
                            ingestion_status["current_file"] = filename

                            # Persistent storage: static/raw/sharded/drive_id/filename
                            raw_dir = get_sharded_path(os.path.join("static", "raw"), drive_id)
                            os.makedirs(raw_dir, exist_ok=True)
                            local_path = os.path.join(raw_dir, filename)

                            if not os.path.exists(local_path):
                                print(f"Downloading {filename}...")
                                download_file(service, drive_id, local_path)

                            process_document(local_path, drive_id, filename, current_db_id, tags=path_tags)

                        ingestion_status["processed_count"] += 1
            except Exception as e:
                error_log("process_recursive", str(e))

        # 1 & 2. Traverse and Process in one go
        process_recursive(folder_id, root_db_id, [root_name])

        # 3. Reconciliation Step
        debug_print("Reconciliation step...")
        all_db_docs = execute_query("SELECT drive_id FROM documents WHERE hidden = false")
        db_drive_ids = [doc['drive_id'] for doc in all_db_docs]

        missing_ids = set(db_drive_ids) - set(processed_drive_ids)

        if missing_ids:
            print(f"Hiding {len(missing_ids)} missing documents...")
            execute_query("UPDATE documents SET hidden = true WHERE drive_id = ANY(%s)", (list(missing_ids),),
                          fetch=False)

        reappeared_ids = set(processed_drive_ids) & set(
            [doc['drive_id'] for doc in execute_query("SELECT drive_id FROM documents WHERE hidden = true")])
        if reappeared_ids:
            print(f"Restoring {len(reappeared_ids)} reappeared documents...")
            execute_query("UPDATE documents SET hidden = false WHERE drive_id = ANY(%s)", (list(reappeared_ids),),
                          fetch=False)

        print("Ingestion complete.")
    finally:
        ingestion_status["is_running"] = False
        ingestion_lock.release()


def run_ingestion(folder_link, mode="Fast"):
    if ingestion_status["is_running"]:
        return False

    # Start ingestion in a background thread
    thread = threading.Thread(target=run_ingestion_thread, args=(folder_link, mode))
    thread.start()
    return True
