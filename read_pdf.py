import os
from typing import Tuple, List

import fitz
import pytesseract
from PIL import Image
from os import path, walk
import pandas as pd
import easyocr
import numpy as np
from datetime import datetime

def traverse_folder(root_folder) -> List[Tuple[str, str]]:
    file_list = []

    for folder_name, sub_folders, filenames in walk(root_folder):
        if folder_name.startswith("pdf\\กองคลัง"):
            continue
        for f in filenames:
            relative_path = str(path.relpath(folder_name, root_folder))
            file_list.append((relative_path, f))

    return file_list


def pdf_to_text(root_dir="pdf",
                pytesseract_exe=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                output_filename="text/summary",
                ocr_engine="tesseract",
                img_dir="img"):
    pytesseract.pytesseract.tesseract_cmd = pytesseract_exe
    reader = easyocr.Reader(['th', 'en'])
    files = traverse_folder(root_dir)
    files = files[0: 3]
    data = []

    for rel_path, filename in files:
        if not os.path.exists(os.path.join(img_dir, rel_path)):
            os.makedirs(os.path.join(img_dir, rel_path))
        with fitz.open(path.join("pdf", rel_path, filename)) as doc:  # open a document
            for i, page in enumerate(doc):
                pix = page.get_pixmap()  # render page to an image
                image_filepath = path.join("img", rel_path, f"{filename}_{i}.png")
                if os.path.exists(image_filepath):
                    print("skipped", filename, i)
                else:
                    pix.save(image_filepath)
                    print(filename, i)
                img_obj = Image.open(image_filepath)
                if ocr_engine == "tesseract":
                    s = pytesseract.image_to_string(img_obj, lang="tha+eng")
                else:
                    s = " ".join(reader.readtext(np.array(img_obj), detail=0, paragraph=True))
                data.append([filename, rel_path, i, s])

    df = pd.DataFrame(data, columns=["filename", "relative_path", "page", "text"])
    df.to_csv(f"{output_filename}_{ocr_engine}.csv")
    print("done")


if __name__ == "__main__":
    start = datetime.now()
    pdf_to_text()
    finish = datetime.now()
    print(start, finish, finish - start)
