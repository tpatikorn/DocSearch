import os

import dotenv
import pandas as pd
import pymupdf
from PIL import Image
from google import genai
from google.genai.types import GenerateContentConfig

dotenv.load_dotenv()

# --- Setup ---
OUTPUT_FOLDER = "text_cleaned"
CONSOLIDATE_FILEPATH = os.path.join(OUTPUT_FOLDER, "ocr_docs.csv")

GEMINI_MODEL = ["gemini-2.5-pro"]  # switch model to different models

if not os.path.exists(CONSOLIDATE_FILEPATH):
    _blank = pd.DataFrame({
        "relative_path": [],
        "filename": [],
        "page": [],
        "text": [],
        "error": []
    })
    _blank.to_csv(CONSOLIDATE_FILEPATH, index=False)

# The client will automatically pick it up.
try:
    client = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini client. Make sure the API key is set: {e}")
    # You might want to exit or handle this error differently
    # For this example, we'll assume the environment is set up.
    # If you need to explicitly set the key: client = genai.Client(api_key="YOUR_API_KEY")
    pass

gemini_call_count = 0


def gemini_ocr(image_path: str) -> str:
    """
    Performs Optical Character Recognition (OCR) on an image file using the
    Gemini API, specifically prompting it for Thai, English, and numeral extraction.

    Args:
        image_path: The file path to the image of the scanned document.

    Returns:
        The extracted text as a string, or an error message if processing fails.
    """
    # 1. Prepare the image and the prompt
    # Open the image using Pillow (PIL)
    _img = Image.open(image_path)

    # The prompt explicitly guides the model to perform OCR and handle
    # the specific languages and numerals (Thai and English).
    prompt = (
        "Extract ALL text from this scanned document image. Extracted as much text in its original "
        "document structure as possible, preserving line breaks and sections if possible while avoiding reciting copyrighted material."
    )

    # 2. Call the Gemini API
    global gemini_call_count
    gemini_call_count += 1

    config = GenerateContentConfig(
        temperature=1,
        response_mime_type="text/plain",

    )

    # We send both the text prompt and the image object (as a list) to the model.
    response = client.models.generate_content(
        model=GEMINI_MODEL[gemini_call_count % len(GEMINI_MODEL)],
        contents=[prompt, _img],
        config=config

    )

    for c in response.candidates:
        print("candidates:", c.finish_reason, c.content, c.safety_ratings, c.citation_metadata)
    # 3. Return the extracted text
    return response.text


# --- Example Usage (Requires a dummy image file) ---
if __name__ == '__main__':
    # --- Create a Dummy Image for Demonstration ---
    # **NOTE**: In a real scenario, replace this path with your actual
    # extracted PDF page image (e.g., 'page_1.png', 'scan_001.jpg').
    consolidated_docs = pd.read_csv(CONSOLIDATE_FILEPATH)
    consolidated_docs = consolidated_docs[consolidated_docs["error"].isna()]  # only ones without errors
    consolidated_docs = consolidated_docs[
        consolidated_docs["text"].str.len() > 10]  # if the old one is shit, try to do it again
    consolidated_docs['page'] = consolidated_docs['page'].astype("int")

    image_root = "img"
    raw_folders = ["pdf"]
    image_filepaths = []
    while raw_folders:
        raw_folder = raw_folders.pop()
        for elt in os.listdir(raw_folder):
            this_filepath = os.path.join(raw_folder, elt)
            if os.path.isdir(this_filepath):
                raw_folders.append(this_filepath)
                os.makedirs(os.path.join(image_root, this_filepath), exist_ok=True)
            elif os.path.splitext(this_filepath)[-1].lower() in [".pdf"]:
                with pymupdf.open(this_filepath) as doc:  # open a document
                    for i, page in enumerate(doc):
                        dst_image_filepath = os.path.join(image_root, f"{this_filepath}_{i + 1:03}.png")
                        if os.path.exists(dst_image_filepath):
                            print("skipped", dst_image_filepath)
                        else:
                            pix = page.get_pixmap(dpi=300)  # render page to an image
                            pix.save(dst_image_filepath)
                            print(dst_image_filepath)
                        image_filepaths.append(dst_image_filepath)

    for image_filepath in image_filepaths:
        relative_path, filename = os.path.split(image_filepath)
        filename = os.path.splitext(filename)[0]
        page = int(filename[filename.rfind("_") + 1:])
        filename = filename[0: filename.rfind("_")]

        done_items = consolidated_docs[
            (consolidated_docs["relative_path"] == relative_path) &
            (consolidated_docs["filename"] == filename) &
            (consolidated_docs["page"] == page)]

        # skipping items that's already been OCR'd
        if len(done_items):
            # print(f"Already done OCR on: {image_filepath}. Skipped")
            continue

        try:
            print(f"Attempting OCR on: {image_filepath}")

            # Run the OCR function
            extracted_text = gemini_ocr(image_filepath)

            ## Display the Results ##
            this_result = pd.DataFrame({
                "relative_path": [relative_path],
                "filename": [filename],
                "page": [page],
                "text": [extracted_text],
                "error": [""]
            })
        except Exception as e:
            this_result = pd.DataFrame({
                "relative_path": [relative_path],
                "filename": [filename],
                "page": [page],
                "text": [""],
                "error": [str(e)]
            })
            print(str(e))
        this_result.to_csv(CONSOLIDATE_FILEPATH, index=False, mode='a', header=False)
