FROM python:3.10-slim
WORKDIR /app

COPY . /app

RUN pip --no-cache-dir install -r requirements.txt
EXPOSE 9696
CMD ["python3", "api.py"]
