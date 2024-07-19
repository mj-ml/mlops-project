import requests

params = {
    "temp": 0,
    "hour": 1,
    "day": 150
}

url = 'http://localhost:9696/predict'
response = requests.post(url, json=params)
print(response.json())
