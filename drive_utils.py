import io
import os

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from db import error_log, debug_print

SERVICE_ACCOUNT_FILE = 'service_key.json'
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']


def get_drive_service():
    if os.path.exists(SERVICE_ACCOUNT_FILE):
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        return build('drive', 'v3', credentials=creds)
    else:
        raise FileNotFoundError(f"{SERVICE_ACCOUNT_FILE} not found. Please provide the service account key file.")


def list_files_in_folder(service, folder_id):
    results = []
    page_token = None
    while True:
        query = f"'{folder_id}' in parents and trashed = false"
        response = service.files().list(q=query,
                                        spaces='drive',
                                        fields='nextPageToken, files(id, name, mimeType, md5Checksum, size)',
                                        pageToken=page_token).execute()
        results.extend(response.get('files', []))
        page_token = response.get('nextPageToken', None)
        if not page_token:
            break
    return results


def download_file(service, file_id, destination):
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(destination, 'wb')
    try:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            debug_print(f"Download {int(status.progress() * 100)}%.")
    except Exception as e:
        error_log(f"Error downloading {file_id}", str(e))
