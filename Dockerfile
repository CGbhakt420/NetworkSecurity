FROM python:3.10-slim
WORKDIR /app
COPY . /app

# RUN apt update -y && apt install awscli -y

RUN apt-get update && pip install -r requirements.txt
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]