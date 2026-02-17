# --- STAGE 1: Builder (Python) ---
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS python-builder

ARG EXTRAS='all'
WORKDIR /build

# Copia solo i file necessari per risolvere le dipendenze
COPY pyproject.toml .
COPY ./tilellm ./tilellm

# Installazione ultra-rapida con uv
# Usiamo --no-cache per non sprecare spazio nel layer del builder
RUN if [ -z "$EXTRAS" ]; then \
    uv pip install --system --no-cache . uvicorn[standard] gunicorn; \
    else \
    uv pip install --system --no-cache ".[$EXTRAS]" uvicorn[standard] gunicorn; \
    fi

# --- STAGE 2: Builder (Node.js) ---
FROM node:16-slim AS node-builder
WORKDIR /usr/src/app
COPY worker/package*.json ./
ARG NPM_TOKEN
RUN if [ "$NPM_TOKEN" ]; then echo "//registry.npmjs.org/:_authToken=$NPM_TOKEN" > .npmrc; fi
RUN npm install --production && rm -f .npmrc
COPY ./worker .

# --- STAGE 3: Final Runtime ---
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Variabili d'ambiente
ENV REDIS_HOST=redis \
    REDIS_URL=redis://redis:6379/0 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONPATH=/tiledesk-llm \
    # Spostiamo i browser in una cartella specifica per evitare che finiscano in /root
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright 

WORKDIR /tiledesk-llm

# 1. Installazione dipendenze di sistema (incluso Node.js per il tuo worker)
RUN apt update && apt install -y --no-install-recommends \
    poppler-utils exiftool curl gnupg ca-certificates \
    && mkdir -p /etc/apt/keyrings \
    && curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg \
    && echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_18.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list \
    && apt update && apt install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# 2. Copia i file dai builder
COPY --from=python-builder /usr/local /usr/local
COPY --from=node-builder /usr/src/app /usr/src/app
COPY . .

# 3. Installazione Playwright e dipendenze usando Python (UV)
# Questo sostituisce "npx playwright install-deps" evitando l'errore npx not found
RUN uv pip install --system --no-cache playwright \
    && playwright install-deps chromium \
    && playwright install chromium \
    && python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng stopwords \
    # Pulizia finale della cache apt che playwright install-deps potrebbe aver ricreato
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 4. Download Modelli
ARG DOWLOADMODEL=false
RUN if [ "$DOWLOADMODEL" != "true" ]; then \
    python -c "from transformers import AutoModelForSequenceClassification; AutoModelForSequenceClassification.from_pretrained('naver/splade-cocondenser-ensembledistil');" && \
    python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2');"; \
    fi

EXPOSE 8000 3009
RUN chmod +x /tiledesk-llm/entrypoint.sh

#ENTRYPOINT ["/tiledesk-llm/entrypoint.sh"]
ENTRYPOINT ["sh","-c","/tiledesk-llm/entrypoint.sh & node /usr/src/app/index.js"]
    