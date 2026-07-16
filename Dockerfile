# Eski halini (muhtemelen python:3.12-slim veya benzeri bir trixie imajıydı) bununla değiştir:
FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x start.sh

EXPOSE 7860

CMD ["./start.sh"]
