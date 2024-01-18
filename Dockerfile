FROM python:3.10-slim
WORKDIR /app
COPY /data /app/data/
COPY /src /app/src/
COPY requirements.txt /app
COPY config.json /app
RUN pip install -r requirements.txt