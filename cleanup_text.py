import json
import os
import time

import dotenv
import google.generativeai as genai
import pandas as pd
import requests
from google.generativeai.types.answer_types import FinishReason
from tqdm import tqdm
import re

dotenv.load_dotenv()

# --- CONFIGURATION ---
MODEL_NAME = "qwen3:8b"
TEXT_FOLDER = "text"
OUTPUT_FOLDER = "text_cleaned"
CSV_FILES = os.listdir(TEXT_FOLDER)  # Paths to CSVs
CONSOLIDATE_FILEPATH = os.path.join(OUTPUT_FOLDER, "cleaned_consolidated_docs.csv")
REDO_EVERYTHING = False

if not os.path.exists(CONSOLIDATE_FILEPATH):
    _blank = pd.DataFrame({
        "relative_path":[],
        "filename":[],
        "page":[],
        "clean_text":[],
        "meta_type":[],
        "meta_subject":[],
        "meta_entities":[],
        "vector_context":[],
        "error":[]
    })
    _blank.to_csv(CONSOLIDATE_FILEPATH, index=False)

# --- 1. DATA PREPARATION ---
print("Loading and merging CSVs...")

# Load all CSVs
dfs = [pd.read_csv(os.path.join(TEXT_FOLDER, f)) for f in CSV_FILES]

# SET THIS FLAG: 'ollama' or 'gemini'
PROVIDER = 'gemini'

# GEMINI SETTINGS
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = ["gemini-2.5-flash", "gemini-2.5-flash-preview-09-2025"] # switch model to different models

# OLLAMA SETTINGS
OLLAMA_MODEL = "qwen3:8b"
OLLAMA_URL = "http://localhost:11434/api/chat"

# --- 2. SETUP BACKENDS ---

if PROVIDER == 'gemini':
    genai.configure(api_key=GEMINI_API_KEY)
    # Configure generation for strict JSON
    generation_config = {
        "temperature": 0.1,
        "response_mime_type": "application/json",
    }
    model_gemini = (
        [genai.GenerativeModel(
            model_name=_model,
            generation_config=generation_config,
            system_instruction="You are a precise Thai Document Editor. Output strictly valid JSON."
        ) for _model in GEMINI_MODEL]
    )

call_count = 0
def call_llm(prompt, model_id=None):
    global call_count
    """Unified wrapper to call either Gemini or Ollama"""
    if model_id is None:
        model_to_use = model_gemini[call_count % len(model_gemini)]
    else:
        model_to_use = model_gemini[model_id]
    call_count += 1
    response = None

    if PROVIDER == 'gemini':
        try:
            # Gemini handles JSON enforcement natively via config
            response = model_to_use.generate_content(prompt)
            return json.loads(response.text)
        except Exception as e:
            print(str(e))
            for c in response.candidates:
                print(">>>>> candidate: ", c.finish_reason, c.token_count)
                print(c.content)
            raise Exception(f"Gemini API Error: {e}")

    elif PROVIDER == 'ollama':
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system",
                     "content": "You are a precise Thai Document Editor. Output strictly valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "format": "json",  # Forces JSON mode in Ollama
                "temperature": 0.1,
                "stream": False
            }
            res = requests.post(OLLAMA_URL, json=payload)
            if res.status_code != 200:
                raise Exception(f"Ollama Error: {res.text}")

            return json.loads(res.json()['message']['content'])
        except Exception as _e:
            raise Exception(f"Ollama Connection Error: {_e}")
    return None


# --- 3. DATA PREPARATION (Using your requested Relative Path Fix) ---
print("Loading and merging CSVs...")
dfs = [pd.read_csv(os.path.join(TEXT_FOLDER, f)) for f in CSV_FILES]

for df in dfs:
    df['relative_path'] = df['relative_path'].astype(str)
    df['filename'] = df['filename'].astype(str)
    df['page'] = df['page'].astype(str)  # Ensure page is string for safe merging

# Rename text columns
for i, df in enumerate(dfs):
    df.rename(columns={'text': f'text_v{i + 1}'}, inplace=True)

# Merge on composite key
merged_df = dfs[0]
for i in range(1, len(dfs)):
    subset = dfs[i][['relative_path', 'filename', 'page', f'text_v{i + 1}']]
    merged_df = pd.merge(merged_df, subset, on=['relative_path', 'filename', 'page'], how='outer')

merged_df = merged_df.fillna("")
print(merged_df.shape)

# a little magic here to skip consolidated rows

if not REDO_EVERYTHING and os.path.exists(CONSOLIDATE_FILEPATH):
    consolidated_docs = pd.read_csv(CONSOLIDATE_FILEPATH)
    # if ['relative_path', 'filename', 'page'] combo is in consolidated_docs, remove it from merged
    # only look at rows where it's not error
    #if "error" in consolidated_docs.columns:
    #    consolidated_docs = consolidated_docs[consolidated_docs["error"].isna()]

    merged_df['page'] = merged_df['page'].astype("int")
    consolidated_docs['page'] = consolidated_docs['page'].astype("int")

    merged_df = merged_df.merge(
        consolidated_docs[['relative_path', 'filename', 'page']],
        on=['relative_path', 'filename', 'page'],
        how='left',
        indicator=True
    )

    # Filter to keep only rows that exist in 'left_only' (merged_df)
    # Then drop the helper column
    merged_df = merged_df[merged_df['_merge'] == 'left_only'].drop(columns=['_merge'])

print(merged_df.shape)


# --- 4. PROCESSING LOOP ---

# wrap additional clean_text() because OCR is dump and can generate shitty characters
def clean_text(text):
    if not text:
        return "Empty"
    # json.dumps escapes quotes and backslashes,
    # and ensures implies surrounding quotes which helps separate the versions.
    return json.dumps(str(text))

def construct_prompt(row):
    """Creates the 'Judge' prompt with 4 versions of the text."""
    return f"""
You are an expert Thai Document Editor. I have processed a single document page using 4 different OCR methods. 
Your job is to combine them into ONE perfect version and extract metadata.

INPUT VERSIONS:
Version 1: {clean_text(row.get('text_v1', '').replace('/.', ' '))}
Version 2: {clean_text(row.get('text_v2', ''))}
Version 3: {clean_text(row.get('text_v3', ''))}
Version 4: {clean_text(row.get('text_v4', ''))}

INSTRUCTIONS:
1. **Consolidate & Fix**: Compare the versions to reconstruct the most likely original text. Fix Thai vowel issues (floating vowels) and spelling errors.
2. **Metadata**: Analyze the content to identify the Document Type, Subject, and Key Entities.
3. **Numbers**: Keep original numbers, but fix format errors (e.g., '1,0 00' -> '1,000').
4. **Fix Common Thai OCR errors**: Fix broken vowels (e.g., 'เ- ก- า' -> 'เกา'). Fix confused numbers (Thai ๑ vs Arabic 1). 
5. **DO NOT translate**: Keep the original language and wording as much as possible and only fix mistakes and typos. 

OUTPUT FORMAT (Strict JSON):
{{
    "clean_text": "The full corrected text of the page...",
    "doc_type": "The type of document (e.g., ระเบียบ,)",
    "subject": "A 1-sentence summary of the page context",
    "entities": "List of names, dates, or organizations found"
}}
"""


error_count = 0

this_result = None
print(f"Starting processing using provider: {PROVIDER.upper()}...")

for index, this_row in tqdm(merged_df.iterrows(), total=merged_df.shape[0]):
    # Skip empty rows
    if all(this_row[f'text_v{i + 1}'] == "" for i in range(len(dfs))):
        continue

    this_result = {
        'relative_path': this_row['relative_path'],
        'filename': this_row['filename'],
        'page': this_row['page'],
        'clean_text': '',
        'meta_type': '',
        'meta_subject': '',
        'meta_entities': '',
        # Metadata string for embedding
        'vector_context': ''
    }

    try:
        new_prompt = construct_prompt(this_row)
        ai_data = call_llm(new_prompt)

        entities = ai_data.get('entities', '')
        if hasattr(ai_data.get('entities'), '__len__'):
            entities = ",".join(entities)

        # Success
        this_result['clean_text'] = ai_data.get('clean_text', '')
        this_result['meta_type'] = ai_data.get('doc_type', '')
        this_result['meta_subject'] = ai_data.get('subject', '')
        this_result['meta_entities'] = entities
        # Metadata string for embedding
        this_result['vector_context'] = f"Type: {ai_data.get('doc_type')} | Subject: {ai_data.get('subject')} | Entities: {entities}"

        # Rate Limit Safety for Gemini (Free tier usually allows 15 RPM, Paid is higher)
        if PROVIDER == 'gemini':
            time.sleep(1)

    except Exception as e:
        print(f"Error processing {this_row['filename']} p{this_row['page']}: {e}")
        # Fallback to raw text
        this_result['error'] = str(e)
        error_count += 1
        time.sleep(6) # sleep on it. It might hit RPM
        if error_count >= 10:
            print("too many error, likely caused by RPD limit reached")
            break
    else:
        error_count = 0

    pd.DataFrame(this_result, index=[0]).to_csv(CONSOLIDATE_FILEPATH, index=False, mode='a', header=False)

