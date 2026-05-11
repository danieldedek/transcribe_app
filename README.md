# Transcribe App

Webová aplikace na platformě Umbrel pro automatický přepis audio souborů (WAV) pomocí tří ASR modelů — Canary, Parakeet a Whisper.

---

## Obsah

- [Přehled funkcí](#přehled-funkcí)
- [Instalace a spuštění pomocí VirtualBoxu](#instalace-a-spuštění-pomocí-virtualboxu)
- [Uživatelská příručka](#uživatelská-příručka)
- [Architektura](#architektura)
- [Testování](#testování)

---

## Přehled funkcí

- Nahrání WAV souboru přes webové rozhraní
- Výběr ze tří ASR modelů: **Canary**, **Parakeet**, **Whisper**
- Nastavení parametrů přepisu (strategie dekódování, beam size, jazyk, FP16…)
- Zobrazení přepisu s volitelnými **časovými razítky** a **mírou spolehlivosti** u modelu Whisper
- Stránkování, vyhledávání a řazení nahraných souborů
- Stažení přepisu jako `.txt`
- Přehrání nahraného souboru přímo v prohlížeči

---

## Instalace a spuštění pomocí VirtualBoxu

1. Stáhněte Umbrel ISO z: `https://github.com/getumbrel/umbrel/wiki/Install-umbrelOS-on-a-Linux-VM`.
2. Vytvořte virtuální stroj a vložte do něho ISO soubor.
3. Pro spuštění rozhraní Umbrel v prohlížeči, do prohlížeče zadejte adresu: `http://umbrel.local`.
4. Otevřete obchod s aplikacemi **App Store**.
5. Vpravo nahoře vedle vyhledávání stiskněte **⋯** (tři tečky).
6. Zvolte možnost **Community App Stores**.
7. Do pole URL vložte: `https://github.com/danieldedek/shop` a stiskněte tlačítko **Add**.
8. Otevřete komunitní obchod stisknutím tlačítka **Open** a zvolte aplikaci **Transcribe App**.
9. Nainstalujte aplikaci stisknutím tlačítka **Install**.
10. Po dokončení instalace můžete aplikaci spustit stisknutím tlačíka **Open**.

---

## Uživatelská příručka

### Nahrání souboru

1. Klikněte na tlačítko **Procházet...** (pole `file`).
2. Vyberte soubor ve formátu `.wav`.
3. Stiskněte **Upload**.
4. Soubor se zobrazí v levém panelu se seznamem nahraných souborů.

### Výběr modelu

V rozbalovacím menu **Select model** vyberte jeden z modelů:

| Model | Popis |
|---|---|
| **Canary** | `nvidia/canary-180m-flash` z NVIDIA NeMo. |
| **Parakeet** | `nvidia/parakeet-tdt-0.6b-v3` z NVIDIA NeMo. |
| **Whisper** | Využívá knihovnu **faster-whisper** (CTranslate2 backend). |

Po výběru klikněte na **Select model** pro potvrzení.

### Parametry přepisu

#### Společné parametry

| Parametr | Popis |
|---|---|
| **Device** | `cpu` nebo `cuda` (GPU) — Whisper tuto volbu nezobrazuje, detekuje automaticky |
| **Strategy** | `greedy` nebo `beam` (přesnější, pomalejší) — Whisper tuto volbu nezobrazuje, používá parametr `temperature` |
| **Beam size** | Šířka paprsku pro beam dekódování (výchozí: 5) |

#### Whisper — specifické parametry

| Parametr | Popis |
|---|---|
| **Model size** | `tiny` / `base` / `small` / `medium` / `large-v3` — čím větší, tím přesnější a pomalejší |
| **Language** | Jazyk audio souboru (`cs` = čeština, `en` = angličtina…) |
| **Temperature** | Náhodnost výstupu (0.0 = deterministický) |
| **Best of** | Počet vzorků při nenulové teplotě |
| **VAD filter** | Přeskočí tiché části pomocí Voice Activity Detection |
| **Condition on previous text** | Podmíní dekódování předchozím textem (lepší kontext) |
| **Show timestamps** | Zobrazí tabulku s časovými značkami segmentů |
| **Show confidence** | Zobrazí míru spolehlivosti každého segmentu |

#### Canary — specifické parametry

| Parametr | Popis |
|---|---|
| **Language** | `en` / `de` / `es` / `fr` |
| **Length penalty** | Penalizace délky výstupu (vyšší = delší přepisy) |
| **FP16** | Poloviční přesnost |
| **Return hypotheses** | Vrátí více hypotéz dekódování |

#### Parakeet — specifické parametry

| Parametr | Popis |
|---|---|
| **FP16** | Poloviční přesnost |
| **Return hypotheses** | Vrátí více hypotéz dekódování |

### Spuštění přepisu

1. V levém panelu klikněte na **Use** u souboru, který chcete přepsat.
2. Aplikace zobrazí spinner s textem *„Transcribing, please wait…"*.
3. Po dokončení se přepis zobrazí v textovém poli.
4. Pokud je zapnuto **Show timestamps**, zobrazí se tabulka segmentů pod textovým polem.

### Stažení přepisu

Pod zobrazeným přepisem klikněte na tlačítko **Download** — stáhne se soubor `transcript.txt`.

### Smazání souboru

V levém panelu klikněte na tlačítko **Delete** u daného souboru. Smazání je nevratné.

### Vyhledávání a řazení

- **Search** — filtruje soubory podle části názvu.
- **Sort by Name / Sort by Date** — přepíná řazení abecedně nebo podle data poslední modifikace (nejnovější první).
- Výsledky jsou stránkované po 5 souborech, lze přecházet tlačítky **Prev** / **Next**.

---

## Architektura

```
transcribe-app/
├── app/
│   ├── main.py          # Flask aplikace, routování, logika přepisu
│   ├── utils.py         # Továrna modelů + cache
│   ├── baseASR.py       # Abstraktní třída BaseASR
│   ├── whisper.py       # Implementace Whisper
│   ├── canary.py        # Implementace Canary
│   ├── parakeet.py      # Implementace Parakeet
│   ├── templates/
│   │   └── index.html   # Jinja2 šablona UI
│   ├── static/
│   │   └── style.css    # Styly
│   └── tests/           # Pytest testy
├── Dockerfile
├── entrypoint.sh        # Nastavení oprávnění + spuštění jako UID 1000
└── .github/workflows/
    └── docker-image.yml # CI/CD pipeline
```

---

## Testování

```bash
# Spusťte jednotkové testy
sudo docker exec -it shop-shop_server_1 pytest app/tests/ -m "unit" -v

# Spusťte integrační testy Flasku
sudo docker exec -it shop-shop_server_1 pytest app/tests/ -m "integration" -v

# Spusťte end-to-end testy
sudo docker exec -it shop-shop_server_1 pytest app/tests/ -m "e2e" -v

# Spusťte inference testy
sudo docker exec -it shop-shop_server_1 pytest app/tests/ -m "inference" -v

# Spusťte testy konkrétního modelu
sudo docker exec -it shop-shop_server_1 pytest app/tests/ -m "whisper" -v
sudo docker exec -it shop-shop_server_1 pytest app/tests/ -m "canary" -v
sudo docker exec -it shop-shop_server_1 pytest app/tests/ -m "parakeet" -v
```

### Struktura testů

| Soubor | Popis |
|---|---|
| `test_whisper.py` | Unit testy inicializace + inference testy |
| `test_canary.py` | Unit testy inicializace + inference testy |
| `test_parakeet.py` | Unit testy inicializace + inference testy |
| `test_utils.py` | Testy cache a továrny modelů |
| `test_integration.py` | HTTP testy Flask routování |
| `conftest.py` | Sdílené fixtures (WAV soubory, Flask klient) |
| `test_config.py` | Konfigurace testů (modely, zařízení, reálné soubory) |
