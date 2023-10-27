import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download("wordnet")
nltk.download("punkt")
nltk.download("stopwords")


def process_text(text: list) -> list:
    py_lemmatizer = WordNetLemmatizer()
    stop = stopwords.words("english")
    exclude = []
    stop_words = [word for word in stop if word not in exclude]

    REPLACE_BY_SPACE_RE = re.compile("[/(){}\[\]\|@,;]")
    BAD_SYMBOLS_RE = re.compile("[^0-9a-z #+_<.*?>]")
    final_list = []
    alphabet = [
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "i",
        "j",
        "k",
        "l",
        "m",
        "n",
        "o",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
    ]
    for sentence in text:
        # Check if the sentence is a missing value
        if isinstance(sentence, str) == False:
            sentence = ""

        filtered_sentence = []

        # Lowercase
        sentence = sentence.lower()
        # Remove leading/trailing whitespace
        sentence = sentence.strip()
        # Remove extra space and tabs
        sentence = re.sub("\s+", " ", sentence)
        # Remove HTML tags/markups:
        sentence = REPLACE_BY_SPACE_RE.sub(
            " ", sentence
        )  # replace REPLACE_BY_SPACE_RE symbols by space in sentence
        sentence = BAD_SYMBOLS_RE.sub(
            "", sentence
        )  # delete symbols which are in BAD_SYMBOLS_RE from text

        for word in word_tokenize(sentence):
            # Check if it is not numeric and its length>2 and not in stop words
            k = 1
            for letter in word:
                if letter not in alphabet:
                    k = 0

            if (
                (not word.isnumeric())
                and (len(word) > 2)
                and (word not in stop_words)
                and (k == 1)
            ):
                # lemmatize and add to filtered list
                filtered_sentence.append(py_lemmatizer.lemmatize(word))

            string = " ".join(filtered_sentence)  # final string of cleaned words

        final_list.append(string)

    return final_list


def get_subjects(entries: list) -> list:
    subjects = []
    for entry in entries:
        subject = []
        if "mech" in entry:
            subject.insert(-1, "mechanical")
        if "aero" in entry:
            subject.insert(-1, "aeronautical")
        if "material" in entry:
            subject.insert(-1, "material")
        # if it mentions mechanical and aeronautical it is most likely just aeronautical.
        if subject == ["aeronautical", "mechanical"]:
            subject = ["aeronautical"]
        subjects.append(subject)
    return subjects


def get_metrics(df, report, model_name, t):
    row = {
        "Model": model_name,
        "Class": "total",
        "Parameter/Metric": "Training Time",
        "Value": round(t, 2),
    }
    df = df._append(row, ignore_index=True)
    for label in report:
        if label == "accuracy":
            row = {
                "Model": model_name,
                "Class": "total",
                "Parameter/Metric": label.title(),
                "Value": round(report[label], 2),
            }
            df = df._append(row, ignore_index=True)
            continue
        for metric in report[label]:
            if metric == "support":
                continue
            row = {
                "Model": model_name,
                "Class": label.title(),
                "Parameter/Metric": metric.title(),
                "Value": round(report[label][metric], 2),
            }
            df = df._append(row, ignore_index=True)
    return df
