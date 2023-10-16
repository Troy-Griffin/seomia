import json
import pandas as pd

if __name__ == "__main__":

    with open('lexicon.json', 'r') as f:
        data = json.load(f)
    df_lexicon = pd.DataFrame(data)

    with open('article.json', 'r') as f:
        data = json.load(f)
    df_articles = pd.DataFrame(data)