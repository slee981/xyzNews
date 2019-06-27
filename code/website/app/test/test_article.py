import requests
import os

KERAS_REST_API_URL = "http://localhost:5000/predict"
ARTICLES_DIR = os.path.abspath("articles")
TEST_DIR = os.getcwd()

# change to articles dir
# and read in all articles
os.chdir(ARTICLES_DIR)
test_articles = [a for a in os.listdir() if ".txt" in a]

for article in test_articles:
    print(f"\n##### Reading {article}")
    with open(article, "r") as f:
        article = f.read()

    payload = {"article": article}
    r = requests.post(KERAS_REST_API_URL, data=payload).json()

    for source in r["predictions"].keys():
        res = r["predictions"]
        print(f"{source}: {round(res[source], 3)}")

# change back to original dir
os.chdir(TEST_DIR)
