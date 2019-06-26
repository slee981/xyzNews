import requests

KERAS_REST_API_URL = "http://localhost:5000/predict"

article = "This is my article"
payload = {"article": article}

r = requests.post(KERAS_REST_API_URL, data=payload).json()

print(r)

