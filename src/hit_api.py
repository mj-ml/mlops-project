import requests

if __name__ == "__main__":
    """
    Example pipeline 
    """
    params = {"temp": 0, "hour": 1, "day": 150}

    url = "http://localhost:9696/alive"
    response = requests.post(url)

    url = 'http://localhost:9696/train'
    response = requests.post(url)

    url = 'http://localhost:9696/predict'
    response = requests.post(url, json=params)
    print(response.json())
