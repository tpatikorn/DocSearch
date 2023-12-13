import fitz
import pytesseract
from PIL import Image
from pythainlp import tokenize
from os import path, listdir, walk
import pandas as pd

root_dir = "pdf"
filename = "ระเบียบมหาวิทยาลัยเทคโนโลยีราชมงคลสุวรรณภูมิ ว่าด้วยการปฐมนิเทศและปัจฉิมนักศึกษา 2565.pdf"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def traverse_folder(root_folder):
    file_list = []

    for folder_name, sub_folders, filenames in walk(root_folder):
        for f in filenames:
            relative_path = path.relpath(folder_name, root_folder)
            file_list.append((path.join(root_folder, relative_path), f))

    return file_list


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
df.to_csv("summary.csv")
print("done")
