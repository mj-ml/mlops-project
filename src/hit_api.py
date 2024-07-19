import requests

params = {
    "temp": 0,
    "hour": 10,
    "day": 40
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=params)
print(response.json())
