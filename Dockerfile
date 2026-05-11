# Autor: Daniel Dedek

FROM python:3.12-slim

# Zakaz vytvareni .pyc souboru a bufferovani stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Nastaveni pracovniho adresare
WORKDIR /app

# Instalace systemovych zavislosti
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# Vytvoreni systemoveho uzivatele s UID 1000
RUN useradd -u 1000 -m appuser

# Instalace Python zavislosti
COPY app/requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Nastaveni opravneni pro pocatecni soubor
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Naslouchani na portu 5000
EXPOSE 5000

ENTRYPOINT ["/entrypoint.sh"]
CMD ["python", "app/main.py"]
