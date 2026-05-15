import hashlib
import io
import os
import re

import pymupdf
import pytesseract
from PIL import Image
from pythainlp.tokenize import word_tokenize

from db import execute_query, debug_print

# Initialize tessaract
pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_CMD')


def get_sharded_path(base_dir, drive_id):
    """Returns a sharded path like base_dir/ab/cd/drive_id/"""
    m = hashlib.md5()
    m.update(drive_id.encode('utf-8'))
    hash_str = m.hexdigest()
    shard1 = hash_str[:2]
    shard2 = hash_str[2:4]
    return os.path.join(base_dir, shard1, shard2, drive_id)


def get_or_create_folder(drive_id, name, parent_id=None):
    existing = execute_query("SELECT id FROM folders WHERE drive_id = %s", (drive_id,))
    if existing:
        # Update name in case it changed
        execute_query("UPDATE folders SET name = %s, parent_id = %s, updated_at = NOW() WHERE id = %s",
                      (name, parent_id, existing[0]['id']), fetch=False)
        return existing[0]['id']

    result = execute_query("INSERT INTO folders (name, parent_id, drive_id) VALUES (%s, %s, %s) RETURNING id",
                           (name, parent_id, drive_id))
    return result[0]['id']


def get_or_create_tag(name):
    existing = execute_query("SELECT id FROM tags WHERE name = %s", (name,))
    if existing:
        return existing[0]['id']

    result = execute_query("INSERT INTO tags (name) VALUES (%s) RETURNING id", (name,))
    return result[0]['id']


def tag_document(document_id, tag_names):
    for name in tag_names:
        tag_id = get_or_create_tag(name)
        execute_query("""
                      INSERT INTO document_tags (document_id, tag_id)
                      VALUES (%s, %s)
                      ON CONFLICT DO NOTHING
                      """, (document_id, tag_id), fetch=False)


def process_document(file_path, drive_id, filename, folder_id, tags=None):
    # 1. Convert to images
    doc = pymupdf.open(file_path)
    page_count = len(doc)

    # Check if document exists
    existing_doc = execute_query("SELECT id FROM documents WHERE drive_id = %s", (drive_id,))

    if existing_doc:
        doc_id = existing_doc[0]['id']
        execute_query(
            "UPDATE documents SET filename = %s, folder_id = %s, page_count = %s, updated_at = NOW() WHERE id = %s",
            (filename, folder_id, page_count, doc_id), fetch=False)
    else:
        result = execute_query(
            "INSERT INTO documents (filename, folder_id, drive_id, page_count) VALUES (%s, %s, %s, %s) RETURNING id",
            (filename, folder_id, drive_id, page_count))
        doc_id = result[0]['id']

    # Tag document if tags provided
    if tags:
        tag_document(doc_id, tags)

    aggregated_tokens = []

    debug_print("finished reading page: ", end="")
    for i, page in enumerate(doc):
        page_number = i + 1
        pix = page.get_pixmap()

        # Save image for preview using sharding
        img_dir = get_sharded_path(os.path.join("static", "previews"), drive_id)
        os.makedirs(img_dir, exist_ok=True)
        img_path = os.path.join(img_dir, f"page_{page_number}.png")
        pix.save(img_path)

        img_data = pix.tobytes("png")
        img_obj = Image.open(io.BytesIO(img_data))

        # OCR - Tesseract
        text_tess = pytesseract.image_to_string(img_obj, lang='tha+eng')
        text_tess = re.sub(r"\n|\s+", " ", text_tess)
        debug_print(i + 1, end=" ")

        combined_text = text_tess

        # Tokenize
        tokens = word_tokenize(combined_text, engine="newmm")
        tokenized_content = " ".join(tokens)
        aggregated_tokens.extend(tokens)

        # Upsert Page
        execute_query("""
                      INSERT INTO pages (document_id, page_number, content, tokenized_content)
                      VALUES (%s, %s, %s, %s)
                      ON CONFLICT (document_id, page_number) DO UPDATE
                          SET content           = EXCLUDED.content,
                              tokenized_content = EXCLUDED.tokenized_content,
                              updated_at        = NOW()
                      """, (doc_id, page_number, combined_text, tokenized_content), fetch=False)

    print("done!")
    # Update document aggregated tokens
    final_aggregated_tokens = " ".join(aggregated_tokens)
    execute_query("UPDATE documents SET aggregated_tokens = %s, updated_at = NOW() WHERE id = %s",
                  (final_aggregated_tokens, doc_id), fetch=False)

    return doc_id
