from typing import List
from pythainlp import tokenize
from flask import Flask, request, jsonify, render_template
import pandas as pd

text_location = "text/summary.csv"
app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


def _search(query: str, text: pd.DataFrame, title_only: bool, use_tokenizer: bool) -> pd.DataFrame:
    if not use_tokenizer:
        if title_only:
            return text[query in text["filename"]]
        else:
            return text[query in text["filename"] |
                        query in text["text"]]
    else:
        terms: List[str] = tokenize.word_tokenize(query)
        terms.append(query)
        if title_only:
            return text[text["filename"].str.contains("|".join(terms))]
        else:
            return text[text["filename"].str.contains("|".join(terms)) |
                        text["text"].str.contains("|".join(terms))]


@app.route("/search")
def search():
    query: str = request.args.get('query')
    text = pd.read_csv(text_location)
    title_only: bool = (request.args.get('title_only') or "false").lower() == "true"
    use_tokenizer: bool = (request.args.get('use_tokenizer') or "true").lower() == "true"
    return _search(query, text, title_only, use_tokenizer).to_json()


@app.route("/home")
def home():
    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)
    print("test")
