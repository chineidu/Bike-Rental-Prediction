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
.PHONY: db-init db-migrate db-upgrade db-downgrade db-current db-history db-reset db-seed

# Initialize database (migrate + seed)
db-init: db-upgrade db-seed
	@echo "✅ Database initialized successfully!"

# Create a new migration
db-migrate:
	@read -p "Enter migration message: " msg; \
	alembic revision --autogenerate -m "$$msg"

# Apply pending migrations
db-upgrade:
	@echo "⬆️  Applying database migrations..."
	alembic upgrade head

# Create database with initial data (roles, etc.)
db-seed:
	@echo "🌱 Seeding database with initial data..."
	uv run -m src.api.setup

# Rollback last migration
db-downgrade:
	@echo "⬇️  Rolling back last migration..."
	alembic downgrade -1

# Show current migration version
db-current:
	@echo "📍 Current database version:"
	alembic current

# Show migration history
db-history:
	@echo "📜 Migration history:"
	alembic history --verbose

# Reset database (WARNING: destructive)
db-reset:
	@echo "⚠️  WARNING: This will drop all tables!"
	@read -p "Are you sure? (y/N): " confirm; \
	if [ "$$confirm" = "y" ]; then \
		alembic downgrade base; \
		echo "✅ Database reset complete"; \
	else \
		echo "❌ Aborted"; \
	fi
