COMPOSE_FILE := ./docker/docker-compose.yml
ENV_FILE := ./docker/.env
PROFILE ?= app-base

.PHONY: up down restart ps logs logs-app logs-redis

up:
	docker compose --env-file $(ENV_FILE) -f $(COMPOSE_FILE) --profile $(PROFILE) up --build -d

down:
	docker compose --env-file $(ENV_FILE) -f $(COMPOSE_FILE) --profile $(PROFILE) down

restart: down up

ps:
	docker compose --env-file $(ENV_FILE) -f $(COMPOSE_FILE) --profile $(PROFILE) ps

logs:
	docker compose --env-file $(ENV_FILE) -f $(COMPOSE_FILE) --profile $(PROFILE) logs -f

logs-app:
	docker compose --env-file $(ENV_FILE) -f $(COMPOSE_FILE) logs -f app-base app-graph app-ocr app-all

logs-redis:
	docker compose --env-file $(ENV_FILE) -f $(COMPOSE_FILE) logs -f redis
