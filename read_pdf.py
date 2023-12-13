import fitz
import pytesseract
from PIL import Image
from os import path, walk
import pandas as pd


def traverse_folder(root_folder):
    file_list = []

    for folder_name, sub_folders, filenames in walk(root_folder):
        for f in filenames:
            relative_path = path.relpath(folder_name, root_folder)
            file_list.append((path.join(root_folder, relative_path), f))

    return file_list


def pdf_to_text(root_dir="pdf",
                pytesseract_exe=r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                output_filename="text/summary.csv"):
    pytesseract.pytesseract.tesseract_cmd = pytesseract_exe
    files = traverse_folder(root_dir)
    print(*files, sep="\n")
    data = []

    for rel_path, filename in files:
        with fitz.open(path.join(rel_path, filename)) as doc:  # open a document
            for i, page in enumerate(doc):
                print(filename, i)
                pix = page.get_pixmap()  # render page to an image
                image_filepath = path.join("img", f"{filename}_{i}.png")
                pix.save(image_filepath)
                s = pytesseract.image_to_string(Image.open(image_filepath), lang="tha")
                data.append([filename, i, s])

    df = pd.DataFrame(data, columns=["filename", "page", "text"])
    df.to_csv(output_filename)
    print("done")


if __name__ == "__main__":
    pdf_to_text()
