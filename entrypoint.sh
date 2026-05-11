#!/bin/sh

# Autor: Daniel Dedek

# Ukonceni skriptu pri chybe 
set -e

# Vytvoreni pracovniho adresare
mkdir -p /app/uploads
chown -R 1000:1000 /app/uploads

# Prepnuti na uzivatele s UID 1000
exec gosu 1000:1000 "$@"
