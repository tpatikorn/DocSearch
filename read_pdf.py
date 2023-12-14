import os

import fitz
import pytesseract
from PIL import Image
from os import path, walk
import pandas as pd


def traverse_folder(root_folder):
    file_list = []

    for folder_name, sub_folders, filenames in walk(root_folder):
        if folder_name.startswith("pdf\\กองคลัง"):
            continue
        for f in filenames:
            relative_path = path.relpath(folder_name, root_folder)
            file_list.append((relative_path, f))

    return file_list


def pdf_to_text(root_dir="pdf",
                pytesseract_exe=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                output_filename="text/summary.csv",
                img_dir="img"):
    pytesseract.pytesseract.tesseract_cmd = pytesseract_exe
    files = traverse_folder(root_dir)
    print(*files, sep="\n")
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
                s = pytesseract.image_to_string(Image.open(image_filepath), lang="tha")
                data.append([filename, rel_path, i, s])

    df = pd.DataFrame(data, columns=["filename", "relative_path", "page", "text"])
    df.to_csv(output_filename)
    print("done")


if __name__ == "__main__":
    pdf_to_text()
