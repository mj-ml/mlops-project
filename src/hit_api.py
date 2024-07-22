import requests

if __name__ == "__main__":
    """
    Example pipeline 
    """
    params = {"temp": 0, "hour": 1, "day": 150}

    url = "http://localhost:9696/alive"
    response = requests.post(url)
    if response.status_code == 200:
        print("run test alive: all good")

    url = 'http://localhost:9696/train'
    response = requests.post(url)
    if response.status_code == 200:
        print("train the models: all good")

    url = 'http://localhost:9696/predict'
    response = requests.post(url, json=params)
    resp = response.json()
    print(f"the predicted load is {resp['load']}")
