FROM python:3.10

WORKDIR /tiledesk-llm

# COPY log_conf.yaml /tiledesk-llm/log_conf.yaml
COPY log_conf.json /tiledesk-llm/log_conf.json
COPY pyproject.toml /tiledesk-llm/pyproject.toml
COPY ./tilellm /tiledesk-llm/tilellm

RUN pip install .
RUN pip install "uvicorn[standard]" gunicorn
# Aggiustare redis
ENV REDIS_HOST=redis
ENV REDIS_URL=redis://redis:6379/0

# Expose the port your FastAPI application uses (modify if needed)
EXPOSE 8000

COPY entrypoint.sh /tiledesk-llm/entrypoint.sh
RUN chmod +x /tiledesk-llm/entrypoint.sh

ENTRYPOINT ["/tiledesk-llm/entrypoint.sh"]

