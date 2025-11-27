import os

import dotenv
import pandas as pd
from PIL import Image
from google import genai
from google.genai.errors import APIError

dotenv.load_dotenv()

# --- Setup ---
OUTPUT_FOLDER = "text_cleaned"
CONSOLIDATE_FILEPATH = os.path.join(OUTPUT_FOLDER, "ocr_docs.csv")

GEMINI_MODEL = ["gemini-2.5-flash", "gemini-2.5-flash-preview-09-2025"]  # switch model to different models

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
    try:
        # Open the image using Pillow (PIL)
        _img = Image.open(image_path)
    except FileNotFoundError:
        return f"Error: Image file not found at {image_path}"
    except Exception as e:
        return f"Error opening image {image_path}: {e}"

    # The prompt explicitly guides the model to perform OCR and handle
    # the specific languages and numerals (Thai and English).
    prompt = (
        "Extract ALL text, including Thai characters, Thai numerals, and English, "
        "from this scanned document image. Present the extracted text in its original "
        "document structure, preserving line breaks and sections."
    )

    # 2. Call the Gemini API
    global gemini_call_count
    gemini_call_count += 1
    try:
        # We send both the text prompt and the image object (as a list) to the model.
        response = client.models.generate_content(
            model=GEMINI_MODEL[gemini_call_count % len(GEMINI_MODEL)],
            contents=[prompt, _img]
        )

        # 3. Return the extracted text
        return response.text

    except APIError as _e:
        return f"Gemini API Error: {_e}"
    except Exception as _e:
        return f"An unexpected error occurred during API call: {_e}"


# --- Example Usage (Requires a dummy image file) ---
if __name__ == '__main__':
    # --- Create a Dummy Image for Demonstration ---
    # **NOTE**: In a real scenario, replace this path with your actual
    # extracted PDF page image (e.g., 'page_1.png', 'scan_001.jpg').
    consolidated_docs = pd.read_csv(CONSOLIDATE_FILEPATH)
    consolidated_docs = consolidated_docs[consolidated_docs["error"].isna()]  # only ones without errors
    consolidated_docs['page'] = consolidated_docs['page'].astype("int")

    image_folders = ["img"]
    image_filepaths = []
    while image_folders:
        image_folder = image_folders.pop()
        for elt in os.listdir(image_folder):
            this_filepath = os.path.join(image_folder, elt)
            if os.path.isdir(this_filepath):
                image_folders.append(this_filepath)
            elif os.path.splitext(this_filepath)[-1].lower() in [".jpg", ".jpeg", ".png"]:
                image_filepaths.append(this_filepath)

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
            print(f"Already done OCR on: {image_filepath}. Skipped")
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
