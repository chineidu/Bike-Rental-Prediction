# ===== Bike Rental Prediction - Makefile =====
.DEFAULT_GOAL := help
.PHONY: help

# Environment Variables
COMPOSE_FILE := docker-compose.yaml
SHELL := /bin/bash
.SHELLFLAGS := -ec

# ========== Pre-commit Management ==========
.PHONY: pre-commit-update pre-commit-install pre-commit-run
# Update pre-commit hooks to latest versions
pre-commit-update:
	prek auto-update

# Install pre-commit hooks
pre-commit-install:
	prek install

# Run pre-commit on all files
pre-commit-run:
	prek run --all-files

# Update and run pre-commit
pre-commit-refresh: pre-commit-update pre-commit-run

# ========== Database Migrations (Alembic) ==========
.PHONY: db-init db-migrate db-upgrade db-downgrade db-current db-history db-reset db-seed db-stamp

# Initialize database (migrate + seed)
db-init: db-upgrade db-seed
	@echo "✅ Database initialized successfully!"

# Create a new migration
db-migrate:
	@read -p "Enter migration message: " msg; \
	uv run alembic revision --autogenerate -m "$$msg"

# Apply pending migrations
db-upgrade:
	@echo "⬆️  Applying database migrations..."
	uv run alembic upgrade head

# Create database with initial data (roles, etc.)
db-seed:
	@echo "🌱 Seeding database with initial data..."
	uv run -m src.api.setup

# Mark database as current without running migrations (for existing databases)
db-stamp:
	@echo "📌 Stamping database with current migration version..."
	uv run alembic stamp head
	@echo "✅ Database marked as up-to-date"

# Rollback last migration
db-downgrade:
	@echo "⬇️  Rolling back last migration..."
	uv run alembic downgrade -1

# Show current migration version
db-current:
	@echo "📍 Current database version:"
	uv run alembic current

# Show migration history
db-history:
	@echo "📜 Migration history:"
	uv run alembic history --verbose

# Reset database (WARNING: destructive)
db-reset:
	@echo "⚠️  WARNING: This will drop all tables!"
	@read -p "Are you sure? (y/N): " confirm; \
	if [ "$$confirm" = "y" ]; then \
		uv run alembic downgrade base; \
		echo "✅ Database reset complete"; \
	else \
		echo "❌ Aborted"; \
	fi

# ========== Docker Compose ==========
.PHONY: compose-up compose-down compose-down-orphans compose-down-volumes compose-logs compose-ps compose-restart compose-build

compose-up: compose-down
	@echo "🚀 Starting Docker Compose services..."
	docker compose -f ${COMPOSE_FILE} up -d
	@echo "✅ Services started. Use 'make compose-logs' to view logs."

compose-build:
	@echo "🔨 Building Docker images..."
	docker compose -f ${COMPOSE_FILE} build
	@echo "✅ Build complete. Use 'make compose-up' to start services."

compose-up-build: compose-down
	@echo "🔨 Building and starting Docker Compose services..."
	docker compose -f ${COMPOSE_FILE} up --build -d
	@echo "✅ Services started with fresh build."

compose-down:
	@echo "🛑 Stopping Docker Compose services..."
	docker compose -f ${COMPOSE_FILE} down
	@echo "✅ Services stopped."

compose-down-orphans:
	@echo "🛑 Stopping services and removing orphans..."
	docker compose -f ${COMPOSE_FILE} down --remove-orphans
	docker image prune -f
	@echo "✅ Cleanup complete."

compose-down-volumes:
	@echo "⚠️  WARNING: This will delete all volumes (data will be lost)!"
	@read -p "Are you sure? (y/N): " confirm; \
	if [ "$$confirm" = "y" ]; then \
		docker compose -f ${COMPOSE_FILE} down --remove-orphans --volumes; \
		docker image prune -f; \
		echo "✅ Services stopped and volumes removed."; \
	else \
		echo "❌ Aborted."; \
	fi

compose-logs:
	@echo "📜 Following logs (Ctrl+C to stop)..."
	docker compose -f ${COMPOSE_FILE} logs -f

compose-ps:
	@echo "📋 Running services:"
	docker compose -f ${COMPOSE_FILE} ps

compose-restart: compose-down compose-up
	@echo "🔄 Services restarted."

help:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  🚲 Bike Rental Prediction - Makefile Commands"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "� QUICK START"
	@echo "  make quickstart           Complete project setup (all-in-one)"
	@echo ""
	@echo "�📦 PRE-COMMIT MANAGEMENT"
	@echo "  make pre-commit-install    Install pre-commit hooks"
	@echo "  make pre-commit-update     Update hooks to latest versions"
	@echo "  make pre-commit-run        Run hooks on all files"
	@echo "  make pre-commit-refresh    Update and run hooks"
	@echo ""
	@echo "🗄️  DATABASE MIGRATIONS (Alembic)"
	@echo "  make db-init              Complete setup (migrations + seed data)"
	@echo "  make db-migrate           Create new migration (autogenerate)"
	@echo "  make db-upgrade           Apply pending migrations"
	@echo "  make db-seed              Seed initial data (roles)"
	@echo "  make db-stamp             Mark DB as current (for existing tables)"
	@echo "  make db-downgrade         Rollback last migration"
	@echo "  make db-current           Show current migration version"
	@echo "  make db-history           Show migration history"
	@echo "  make db-reset             Drop all tables (⚠️  destructive)"
	@echo ""
	@echo "🐳 DOCKER COMPOSE"
	@echo "  make compose-up           Start services (fast, uses cache)"
	@echo "  make compose-build        Build images only"
	@echo "  make compose-up-build     Build and start services (slower)"
	@echo "  make compose-down         Stop services"
	@echo "  make compose-down-orphans Stop services and remove orphans"
	@echo "  make compose-down-volumes Stop services and remove volumes (⚠️  destructive)"
	@echo "  make compose-logs         Follow all service logs"
	@echo "  make compose-ps           Show running services"
	@echo "  make compose-restart      Restart all services"
	@echo ""
	@echo "🔧 DEVELOPMENT"
	@echo "  make dev-setup            Complete development setup"
	@echo "  make api-run              Run FastAPI application locally"
	@echo "  make test                 Run tests"
	@echo "  make lint                 Run linters"
	@echo "  make format               Format code"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""

# ========== Development ==========
.PHONY: dev-setup api-run test lint format clean

dev-setup: pre-commit-install
	@echo "🔧 Setting up development environment..."
	@if ! command -v uv &> /dev/null; then \
		echo "❌ 'uv' not found. Please install it: https://docs.astral.sh/uv/"; \
		exit 1; \
	fi
	uv sync
	@echo "✅ Development environment ready!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Copy .env.example to .env and configure"
	@echo "  2. Run 'make compose-up' to start services"
	@echo "  3. Run 'make db-init' to initialize database"

api-run: db-init
	@echo "🚀 Starting FastAPI application..."
	uv run -m src.api.app

test:
	@echo "🧪 Running tests..."
	uv run pytest

lint:
	@echo "🔍 Running linters..."
	uv run ruff check .

format:
	@echo "✨ Formatting code..."
	uv run ruff check --fix .
	uv run ruff format .

clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "✅ Cleanup complete!"

# ========== Quick Start ==========
.PHONY: quickstart

quickstart:
	@echo "🚀 Quick Start - Setting up everything..."
	@echo ""
	@echo "Step 1/5: Setting up development environment..."
	@$(MAKE) dev-setup
	@echo ""
	@echo "Step 2/5: Starting Docker services..."
	@$(MAKE) compose-up
	@echo ""
	@echo "Step 3/5: Waiting for services to be ready..."
	@sleep 5
	@echo ""
	@echo "Step 4/5: Initializing database..."
	@$(MAKE) db-init
	@echo ""
	@echo "Step 5/5: Verifying setup..."
	@$(MAKE) compose-ps
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "✅ Quick start complete!"
	@echo ""
	@echo "Services running:"
	@echo "  • MLflow UI:     http://localhost:5001"
	@echo "  • FastAPI:       http://localhost:8000"
	@echo "  • API Docs:      http://localhost:8000/docs"
	@echo "  • Airflow UI:    http://localhost:8080"
	@echo ""
	@echo "Next steps:"
	@echo "  • View logs:     make compose-logs"
	@echo "  • Run API:       make api-run"
	@echo "  • Run tests:     make test"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
