from pythainlp.tokenize import word_tokenize

from db import execute_query


def get_common_tags():
    sql = """
        SELECT t.id, t.name, COUNT(dt.document_id) as doc_count
        FROM tags t
        JOIN document_tags dt ON t.id = dt.tag_id
        GROUP BY t.id, t.name
        ORDER BY doc_count DESC
        LIMIT 20
    """
    return execute_query(sql)


def search_documents(query_text, tag_ids=None, limit =20):
    # 1. Tokenize query
    tokens = word_tokenize(query_text, engine="newmm", keep_whitespace=False)

    # Phase 1: Document Match (AND)
    and_query = " & ".join(tokens)

    tag_filter_sql = ""
    params = [and_query]
    
    if tag_ids:
        tag_filter_sql = "AND d.id IN (SELECT document_id FROM document_tags WHERE tag_id = ANY(%s) GROUP BY document_id HAVING COUNT(DISTINCT tag_id) = array_length(%s, 1))"
        params.append(tag_ids)
        params.append(tag_ids)

    params.append(str(limit))

    doc_sql = f"""
        WITH RECURSIVE folder_paths AS (
            SELECT id, name, CAST(name AS text) as full_path, parent_id
            FROM folders
            WHERE parent_id IS NULL
            UNION ALL
            SELECT f.id, f.name, fp.full_path || ' / ' || f.name, f.parent_id
            FROM folders f
            JOIN folder_paths fp ON f.parent_id = fp.id
        )
        SELECT d.id, d.filename, d.drive_id, fp.full_path as folder_name
        FROM documents d
        JOIN folder_paths fp ON d.folder_id = fp.id
        WHERE d.search_vector @@ to_tsquery('simple', %s)
          AND d.hidden = false
          {tag_filter_sql}
        LIMIT %s
    """
    matched_docs = execute_query(doc_sql, tuple(params))

    if not matched_docs:
        return []

    doc_ids = [doc['id'] for doc in matched_docs]

    # Phase 2: Page Match (OR)
    or_query = " | ".join(tokens)

    page_sql = """
               SELECT p.document_id, p.page_number, p.content
               FROM pages p
               WHERE p.document_id = ANY (%s)
                 AND p.search_vector @@ to_tsquery('simple', %s)
               ORDER BY p.document_id, p.page_number \
               """
    matched_pages = execute_query(page_sql, (doc_ids, or_query))

    # Combine results
    results = []
    for doc in matched_docs:
        this_result = {
            "filename": doc['filename'],
            "folder_path": doc['folder_name'],
            "drive_id": doc['drive_id'],
            "pages":[]}
        doc_pages = [p for p in matched_pages if p['document_id'] == doc['id']]
        for p in doc_pages:
            this_result['pages'].append({
                "page_number": p['page_number'],
                "content_snippet": p['content'][:200] + "..."  # Snippet for preview
            })
        results.append(this_result)

    return results
