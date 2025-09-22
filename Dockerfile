# syntax=docker/dockerfile:1
FROM python:3.12-slim

# System deps (psycopg2-binary usually needs none; psycopg sometimes needs libpq)
# RUN apt-get update && apt-get install -y --no-install-recommends libpq5 && rm -rf /var/lib/apt/lists/*

# Ensure consistent stdout/stderr and no .pyc files (good for containers)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

#Workdir for app
WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy source after deps for faster rebuilds
COPY . /app

# Dev-friendly reload; fine for local dev
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
