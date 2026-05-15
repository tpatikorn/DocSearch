# System Requirements Specification (SRS): Flask Document Search

1. Project Overview

   A secure, high-performance document search web application optimized for the Thai language. The system ingests nested
   files from Google Drive, converts all pages to images, performs OCR, tokenizes the Thai text, and stores the data in
   PostgreSQL for sub-second Full-Text Search. The application utilizes a Python/Flask backend and a lightweight HTML/JS
   frontend.

2. Technical Stack
    - Backend: Python 3.10+, Flask.
    - Database: PostgreSQL 15+ (interacting via psycopg2 or SQLAlchemy Core).
    - Authentication: Google OAuth 2.0 (Strict login enforcement).
    - Document Processing Pipeline:
        - Fetch: Google Drive API.
        - Conversion: pdf2image, python-docx (or LibreOffice headless for conversion).
        - OCR: pytesseract (Tesseract) and easyocr.
    - NLP/Tokenization: PyThaiNLP.
    - Frontend: HTML5, standard CSS/JS (Jinja2 templates), lightweight UI (no complex JS frameworks required).

3. Core Modules & Business Logic

   3.1. Module A: Authentication & Access

   Strict Access: All routes (except the login page) must be protected by Google OAuth.
   Users must log in via Google. Upon successful login, verify the user against the users table. If they don't exist,
   create a generic 'viewer' record.

   3.2. Module B: The Data Ingestion Engine (Admin Only)

   This process is triggered when an admin provides a Google Drive folder link and selects an execution mode ("Fast"
   or "Full"). The process must be idempotent (safe to run multiple times without creating duplicate records).
    - Execution Modes:
        - Fast Mode: Skips OCR and Tokenization for documents that already exist in the database (matched by Google
          Drive ID) IF the filename and the total page count metadata from Google Drive have not changed.
        - Full Mode: Forces a complete re-download, OCR, and Tokenization of all files, updating existing records in the
          database.

    - The Pipeline Flow:

        - Traversal & Manifest Creation: Recursively traverse the Google Drive folder. Build a manifest of all current
          files, capturing their drive_id, filename, and metadata (like page count).
        - Conditional Processing (The Loop): For each file in the manifest:
        - Check DB: Does this drive_id exist?
        - Fast Mode Check: If Fast Mode is ON, and the DB record matches the Drive's filename and page count, SKIP
          processing.
        - Download & OCR: If Full Mode is ON, or if the file is new/changed, download the file. Convert pages to local
          images (overwriting old images if they exist). Run Tesseract (tha + tha+eng) and EasyOCR.
        - Tokenize: Pass text through PyThaiNLP.
        - Upsert Database: Insert new records, or UPDATE existing pages and documents tables with the fresh
          tokenized_content and aggregated_tokens.

    - The Reconciliation Step (Auto-Hiding Missing Docs):

   Once the loop finishes, the backend must compare the list of drive_ids found in this Drive traversal against the
   drive_ids currently stored in the database for this folder structure.

   If a document exists in the database but was NOT found in the current Google Drive fetch, execute an UPDATE to set
   hidden = true for that document. (Do NOT delete it).

    - Optional: If a previously hidden document reappears in the Drive fetch, set hidden = false.

3.3. Module C: Search API

When a user submits a search query:

- Tokenize Input: Pass the user's raw search query through PyThaiNLP and join with & (AND).

    - Phase 1 (Document Match): Query the documents table using search_vector @@ to_tsquery('simple', '
      tokenized&search&query').
    - Phase 2 (Page Match): For the matched documents, query the pages table using search_vector @@ to_tsquery('
      simple', ' tokenized|search|query') (using OR) to find exactly which pages contain the hits.

3.4. Module D: Frontend UI

The UI is based on the existing home.html prototype vibe but streamlined.

- Login View: A simple page with a "Sign in with Google" button.
- Search View: Centered search bar and submit button. (No extra toggles needed).
- Results View: Displays a list of search results. Each result item MUST display:
- Order Index: (1, 2, 3...)
- Folder Path: (e.g., HR_Docs / 2024 / Q1)
- Document Name: The original filename.
- Page Number: Which page the text was found on.
- Page Image Preview: A button/thumbnail that, when clicked, opens a pop-in/modal displaying the locally stored
  extracted image of that specific page.
- Original Link: A hyperlink out to the original document on Google Drive (drive_id).

4. AI Coding Agent Instructions (Rules of Engagement)

    - Database Schema: Do NOT use ORMs (like SQLAlchemy ORM models) to auto-generate or migrate the schema. The database
      relies on advanced PostgreSQL features (tsvector, generated columns). Rely on the provided schema.sql file for
      database structure. Use parameterized raw SQL or a lightweight query builder for execution.
    - Prototype Reference: Review the user's existing prototype (read_pdf.py and home.html) to understand the desired
      simplicity and "vibe," but upgrade the logic to strictly follow this SRS (specifically migrating from local folder
      traversal to Google Drive API traversal).
    - Error Handling in OCR: OCR and file conversion can be brittle. Wrap Module B in robust try/except blocks. If one
      page fails, log the error but do not crash the entire ingestion pipeline.
    - Concurrency: Consider using Python ThreadPoolExecutor or multiprocessing for the image conversion and OCR steps,
      as these will be highly CPU-bound.