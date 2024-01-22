FROM python:3.10-slim
COPY . /app/
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 5001
CMD [ "python","app.py" ]