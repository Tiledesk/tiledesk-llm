# --- STAGE 1: Builder (Python) ---
FROM python:3.12-slim AS python-builder

ARG EXTRAS='all'

WORKDIR /build
RUN apt update && apt install -y --no-install-recommends \
    gcc g++ libffi-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY ./tilellm ./tilellm
COPY log_conf.json .

# Installazione dipendenze in una cartella locale per facile copia
RUN if [ -z "$EXTRAS" ]; then \
    pip install --no-cache-dir --prefix=/install .; \
    else \
    pip install --no-cache-dir --prefix=/install ".[$EXTRAS]"; \
    fi
RUN pip install --no-cache-dir --prefix=/install "uvicorn[standard]" gunicorn

# --- STAGE 2: Builder (Node.js) ---
FROM node:16-slim AS node-builder
WORKDIR /usr/src/app
COPY worker/package*.json ./
ARG NPM_TOKEN
# Gestione token se presente
RUN if [ "$NPM_TOKEN" ]; then echo "//registry.npmjs.org/:_authToken=$NPM_TOKEN" > .npmrc; fi
RUN npm install --production && rm -f .npmrc
COPY ./worker .

# --- STAGE 3: Final Runtime ---
FROM python:3.12-slim

ARG EXTRAS=""

# Variabili d'ambiente
ENV REDIS_HOST=redis \
    REDIS_URL=redis://redis:6379/0 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/tiledesk-llm \
    PATH="/root/.local/bin:$PATH"

WORKDIR /tiledesk-llm

# Installazione dipendenze di sistema minime
RUN apt update && apt install -y --no-install-recommends \
    poppler-utils exiftool curl \
    && curl -sL https://deb.nodesource.com/setup_16.x | bash - \
    && apt install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copia le librerie Python installate nel builder (solo bin e lib, include non necessario a runtime)
COPY --from=python-builder /install/bin /usr/local/bin
COPY --from=python-builder /install/lib /usr/local/lib
# Copia l'app Node.js
COPY --from=node-builder /usr/src/app /usr/src/app
# Copia il codice sorgente
COPY . .



# NLTK e Playwright (solo Chromium per risparmiare spazio)
RUN python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng stopwords

# Installazione dipendenze minime per Chromium headless (senza X11 per risparmiare spazio)
RUN apt update && apt install -y --no-install-recommends \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2 \
    libatspi2.0-0 \
    libxshmfence1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install playwright \
    && playwright install chromium --without-deps

# Download Modelli (condizionale)
ARG DOWLOADMODEL=false
RUN if [ "$DOWLOADMODEL" != "true" ]; then \
    python -c "from transformers import AutoModelForSequenceClassification; AutoModelForSequenceClassification.from_pretrained('naver/splade-cocondenser-ensembledistil');" && \
    python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2');"; \
    fi

EXPOSE 8000 3009
RUN chmod +x /tiledesk-llm/entrypoint.sh

ENTRYPOINT ["/tiledesk-llm/entrypoint.sh"]
    