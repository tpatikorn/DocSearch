import re

from pythainlp import word_tokenize

from db import execute_query, error_log

if __name__ == "__main__":
    pages = execute_query("SELECT * FROM pages ORDER BY document_id, page_number;")
    this_doc_id = pages[0]["document_id"]
    aggregated_tokens = []
    for page in pages:
        try:
            if this_doc_id != page["document_id"]:
                # if it's not the same document_id, then it means the document is done, update it and move on
                print(f"updating {this_doc_id}")
                final_agg_tokens = " ".join(aggregated_tokens)
                execute_query("UPDATE documents SET aggregated_tokens = %s WHERE id = %s",
                              (final_agg_tokens, this_doc_id,), fetch=False)
                this_doc_id = page["document_id"]
            # if it's still the same document
            doc_text = page['content']
            doc_text = re.sub(r"\s+\u0e32", "\u0e32", doc_text) # fix spaces before "า"
            doc_text = re.sub(r"\s+\u0e4d\s+\u0e32", "\u0e33", doc_text) # fix broken "ํา" -> "ำ"
            doc_text = re.sub(r'[!@#$^&()_=|{}\[\];:\'"<>,.M/+\-*]', ' ', doc_text) # replace special characters with spaces
            doc_text = re.sub(r'[\n\r]', '', doc_text) # remove all newlines
            doc_text = re.sub(r'[\s+]', ' ', doc_text) # combine consecutive spaces
            tokens = word_tokenize(doc_text, engine="newmm", keep_whitespace=False)
            tokenized_content = " ".join(set(tokens))
            aggregated_tokens.extend(tokens)
            execute_query("UPDATE pages SET tokenized_content = %s WHERE document_id = %s AND page_number = %s",
                          (tokenized_content, this_doc_id, page["page_number"]), fetch=False)
        except Exception as e:
            error_log(f"redo_tokenization{page['document_id']} {page['page_number']}", str(e))
