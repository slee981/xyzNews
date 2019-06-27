import requests

KERAS_REST_API_URL = "http://localhost:5000/predict"

ARTICLE = "article_huff.txt"

with open(ARTICLE, "r") as f:
    article = f.read()
payload = {"article": article}

r = requests.post(KERAS_REST_API_URL, data=payload).json()

for source in r["predictions"].keys():
    res = r["predictions"]
    print(f"{source}: {round(res[source], 3)}")
