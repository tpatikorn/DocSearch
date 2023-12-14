import os
from typing import List
from pythainlp import tokenize
from flask import Flask, request, render_template, url_for, send_from_directory
import pandas as pd

text_location = "text/summary.csv"
app = Flask(__name__)


def _search(query: str, text: pd.DataFrame, title_only: bool, use_tokenizer: bool) -> pd.DataFrame:
    if not use_tokenizer:
        if title_only:
            return text[query in text["filename"]]
        else:
            return text[query in text["filename"] |
                        query in text["text"]]
    else:
        terms: List[str] = tokenize.word_tokenize(query)
        if title_only:
            return text[text["filename"].apply(lambda s: all(term in str(s) for term in terms))]
        else:
            for index, row in text.iterrows():
                if type(row['text']) != str:
                    print(index, type(row['text']))
            return text[(text["filename"].apply(lambda s: all(term in str(s) for term in terms))) |
                        (text["text"].apply(lambda s: all(term in str(s) for term in terms)))]


@app.route("/search")
def search():
    query: str = request.args.get('query')
    text = pd.read_csv(text_location, dtype={"filename": "string", "text": "string"})
    title_only: bool = (request.args.get('title_only') or "false").lower() == "true"
    use_tokenizer: bool = (request.args.get('use_tokenizer') or "true").lower() == "true"
    aggregate : bool = (request.args.get('aggregate') or "true").lower() == "false"
    text['text'] = text['filename'].astype(str) + ' ' + text['text']
    if aggregate:
        text = text.groupby(['filename', 'relative_path']).agg({'text': lambda x: ' '.join(x)}).reset_index()
        text['page'] = 0

    return _search(query, text, title_only, use_tokenizer).to_json()


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/fetch")
def fetch_content():
    content_type: str = request.args.get('content_type')
    filename: str = request.args.get('filename')
    relative_path: str = request.args.get('relative_path')
    return send_from_directory(os.path.join(content_type, relative_path), filename)


if __name__ == "__main__":
    app.run(debug=True)
    print("test")
