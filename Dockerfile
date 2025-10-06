# STAGE 1: LLM
FROM python:3.12

ARG IMAGICLE=false

WORKDIR /tiledesk-llm

# COPY log_conf.yaml /tiledesk-llm/log_conf.yaml
COPY log_conf.json /tiledesk-llm/log_conf.json
COPY pyproject.toml /tiledesk-llm/pyproject.toml
COPY ./tilellm /tiledesk-llm/tilellm
# RUN pip install pytest-playwright
# RUN playwright install chromium
# RUN playwright install-deps chromium

RUN echo "Installazione progetto..." && \
    pip install . && \
    echo "Pulizia cache pip..." && \
    pip cache purge

RUN pip install "uvicorn[standard]" gunicorn
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader punkt_tab
RUN python -m nltk.downloader averaged_perceptron_tagger
RUN python -m nltk.downloader averaged_perceptron_tagger_eng
RUN python -m nltk.downloader stopwords
RUN playwright install chromium
RUN playwright install-deps chromium

RUN if [ "$IMAGICLE" = "true" ]; then \
        echo "IMAGICLE è TRUE: Disinstallo torch e salto il download dei modelli."; \
        # Disinstalla torch (usando -y per confermare) e pulisci la cache residua
        # pip uninstall -y torch; \
    else \
        echo "IMAGICLE è FALSE o non impostata: Eseguo download modelli."; \
        # Esegue il download del modello 1 (SPLADE)
        python -c "from transformers import AutoModelForSequenceClassification; AutoModelForSequenceClassification.from_pretrained('naver/splade-cocondenser-ensembledistil');" && \
        # Esegue il download del modello 2 (CrossEncoder)
        python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2');" ; \
    fi
# RUN python -c "from transformers import AutoModelForSequenceClassification; model = AutoModelForSequenceClassification.from_pretrained('BAAI/bge-m3');"
# RUN python -c "from transformers import AutoModelForSequenceClassification; model = AutoModelForSequenceClassification.from_pretrained('naver/splade-cocondenser-ensembledistil');"
# RUN python -c "from sentence_transformers import CrossEncoder; model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2');"

# RUN pip cache purge
# Aggiustare redis
ENV REDIS_HOST=redis
ENV REDIS_URL=redis://redis:6379/0
ENV TOKENIZERS_PARALLELISM=false

# Expose the port your FastAPI application uses (modify if needed)
EXPOSE 8000

COPY entrypoint.sh /tiledesk-llm/entrypoint.sh
RUN chmod +x /tiledesk-llm/entrypoint.sh

# STAGE 2: WORKER

RUN curl -sL https://deb.nodesource.com/setup_16.x | bash -
RUN apt-get install -y nodejs

####
RUN npm install -g npm@8.x.x

#RUN sed -i 's/stable\/updates/stable-security\/updates/' /etc/apt/sources.list

RUN apt-get update

# Create app directory
WORKDIR /usr/src/app
COPY ./worker .


ARG NPM_TOKEN

RUN if [ "$NPM_TOKEN" ]; \
    then RUN COPY .npmrc_ .npmrc \
    else export SOMEVAR=world; \
    fi


# Install app dependencies
# A wildcard is used to ensure both package.json AND package-lock.json are copied
# where available (npm@5+)
COPY worker/package*.json ./

RUN npm install --production



RUN rm -f .npmrc

# Bundle app source
#COPY . .

WORKDIR /tiledesk-llm

EXPOSE 3009

ENTRYPOINT ["sh","-c","/tiledesk-llm/entrypoint.sh & node /usr/src/app/index.js"]
