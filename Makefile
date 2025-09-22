# Variables
DB_HOST := localhost
DB_PORT := 5432
DB_USER := app
DB_NAME := appdb

.PHONY: up down logs-db psql

up:            ## Start the stack
	@docker compose up -d

down:          ## Stop containers (keeps data)
	@docker compose down

logs-db:       ## Tail Postgres logs
	@docker compose logs -f db

psql:          ## Open psql from host
	@psql -h $(DB_HOST) -p $(DB_PORT) -U $(DB_USER) -d $(DB_NAME)