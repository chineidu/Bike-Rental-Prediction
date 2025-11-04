# Database Migration Guide with Alembic

This guide explains how to use Alembic for database migrations without breaking your MLflow database.

## Table of Contents
<!-- TOC -->

- [Database Migration Guide with Alembic](#database-migration-guide-with-alembic)
  - [Table of Contents](#table-of-contents)
  - [Architecture Overview](#architecture-overview)
  - [Initial Alembic Setup First Time - Already Done](#initial-alembic-setup-first-time---already-done)
    - [What Was Done](#what-was-done)
      - [Step 1: Install Alembic](#step-1-install-alembic)
      - [Step 2: Initialize Alembic](#step-2-initialize-alembic)
      - [Step 3: Configured alembic/env.py](#step-3-configured-alembicenvpy)
      - [Step 4: Updated alembic.ini](#step-4-updated-alembicini)
      - [Step 5: Created database init script](#step-5-created-database-init-script)
      - [Step 6: Created initial migration](#step-6-created-initial-migration)
      - [Step 7: Created seed script src/api/setup.py](#step-7-created-seed-script-srcapisetuppy)
  - [Setup](#setup)
    - [Step 0. Configure Environment Variables Optional](#step-0-configure-environment-variables-optional)
    - [Step 1. Start Fresh Database](#step-1-start-fresh-database)
    - [Step 2. Initialize Alembic First Time Only](#step-2-initialize-alembic-first-time-only)
    - [Step 3. Apply Migrations and Seed Data](#step-3-apply-migrations-and-seed-data)
  - [Daily Workflow](#daily-workflow)
    - [Adding a New Table or Column](#adding-a-new-table-or-column)
      - [Step 1. Modify your models in src/db/models.py](#step-1-modify-your-models-in-srcdbmodelspy)
      - [Step 2. Generate migration](#step-2-generate-migration)
      - [Step 3. Review the generated migration in alembic/versions/](#step-3-review-the-generated-migration-in-alembicversions)
      - [Step 4. Apply migration](#step-4-apply-migration)
    - [Checking Migration Status](#checking-migration-status)
    - [Rolling Back a Migration](#rolling-back-a-migration)
  - [Common Commands](#common-commands)
  - [Manual Alembic Commands](#manual-alembic-commands)
  - [Troubleshooting](#troubleshooting)
    - [Problem: "Target database is not up to date"](#problem-target-database-is-not-up-to-date)
    - [Problem: "Alembic can't find my models"](#problem-alembic-cant-find-my-models)
    - [Problem: "Migration conflicts with existing data"](#problem-migration-conflicts-with-existing-data)
    - [Problem: "Want to reset everything"](#problem-want-to-reset-everything)
  - [Database Connection Info](#database-connection-info)
    - [For SQLTools VS Code](#for-sqltools-vs-code)
    - [For Docker Services](#for-docker-services)
  - [Seeding Initial Data](#seeding-initial-data)
    - [Why Separate Schema and Data?](#why-separate-schema-and-data)
    - [Seeding Workflow](#seeding-workflow)
    - [What Gets Seeded?](#what-gets-seeded)
    - [When to Re-run Seeding?](#when-to-re-run-seeding)
    - [Adding Custom Seed Data](#adding-custom-seed-data)
  - [Best Practices](#best-practices)
  - [Safety Features](#safety-features)
  - [Integration with FastAPI](#integration-with-fastapi)
  - [Example Migration Workflow](#example-migration-workflow)
  - [References](#references)

<!-- /TOC -->
## Architecture Overview

```text
PostgreSQL Instance (mlflow-db:5432)
├── ${POSTGRES_DB} (MLflow tracking database - default: mlflow)
│   └── Managed by MLflow (don't touch!)
└── ${API_DB_NAME} (Your API database - default: bikerental_api)
    └── Managed by Alembic (user tables, auth, etc.)
```

**Key Points:**

- **MLflow DB**: Separate database (configurable via `POSTGRES_DB` env var) managed by MLflow itself
- **API DB**: Separate database (configurable via `API_DB_NAME` env var) managed by Alembic
- Both run on the same PostgreSQL instance
- No conflicts because they use different databases
- Database names are configurable in `.env` file

## Initial Alembic Setup (First Time - Already Done)

**⚠️ Important:** This section documents how Alembic was initially configured in this project for reproducibility. **You don't need to run these commands** unless you're:

- Setting up Alembic in a completely new project
- Understanding how this project was bootstrapped
- Recreating the setup from scratch

If you're just using this project, skip to [Setup](#setup) section.

---

### What Was Done

These are the exact commands and configurations that were used to set up Alembic from scratch:

#### Step 1: Install Alembic

```bash
# Added to pyproject.toml dependencies
uv add alembic
```

#### Step 2: Initialize Alembic

```bash
# Created alembic/ directory and alembic.ini config
alembic init alembic
```

This created:

- `alembic.ini` - Main configuration file
- `alembic/` directory with:
  - `env.py` - Environment configuration
  - `script.py.mako` - Migration template
  - `versions/` - Directory for migration files

#### Step 3: Configured `alembic/env.py`

Modified to use the centralized `app_settings` configuration instead of environment variables:

```python
from src.config import app_settings
from src.db.models import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# =============================================================
# ==================== Add DB Config ==========================
# =============================================================
config.set_main_option(
    "sqlalchemy.url",
    f"postgresql+psycopg2://{app_settings.POSTGRES_USER}:{app_settings.POSTGRES_PASSWORD.get_secret_value()}"
    f"@{app_settings.POSTGRES_HOST}:{app_settings.POSTGRES_PORT}/{app_settings.API_DB_NAME}",
)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata
```

**Key improvements:**

- **Simplified configuration**: Uses centralized `app_settings` instead of duplicating environment variable logic
- **Consistent with project**: Leverages the same configuration system used throughout the application
- **Cleaner code**: Removed the `get_url()` function and environment variable parsing
- **Better maintainability**: Database configuration changes only need to be made in one place (`app_settings`)

#### Step 4: Updated `alembic.ini`

```ini
# Set script location
script_location = alembic

# Database URL will be set programmatically in env.py
# sqlalchemy.url = postgresql://user:pass@localhost/dbname
```

#### Step 5: Created database init script

Created `docker/init-databases.sh` to automatically create both databases:

```bash
#!/bin/bash
set -e

API_DB_NAME="${API_DB_NAME:-bikerental_api}"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<EOSQL
    CREATE DATABASE "$API_DB_NAME";
    GRANT ALL PRIVILEGES ON DATABASE "$API_DB_NAME" TO $POSTGRES_USER;
EOSQL
```

#### Step 6: Created initial migration

```bash
# Generate first migration from existing models
make db-migrate
# Message: "Initial schema with users and roles tables"

# Apply migration
make db-upgrade
```

#### Step 7: Created seed script `src/api/setup.py`

Script to populate initial roles after migrations run.

## Setup

### Step 0. Configure Environment Variables (Optional)

You can customize database names in `.env`:

```bash
# PostgreSQL credentials
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=mlflow
POSTGRES_DB=mlflow              # MLflow database name

# API database configuration
API_DB_NAME=bikerental_api      # Your API database name (change if needed)
API_DB_USER=mlflow
API_DB_PASSWORD=mlflow
```

**Note:** If you change `API_DB_NAME`, update it before running `docker-compose up` for the first time.

### Step 1. Start Fresh Database

```bash
# Stop containers and remove volumes
docker-compose down -v

# Start services
docker-compose up -d

# Wait for services to be healthy
docker-compose ps
```

The init script (`docker/init-databases.sh`) will automatically create:

- `${POSTGRES_DB}` database for MLflow
- `${API_DB_NAME}` database for your API

### Step 2. Initialize Alembic (First Time Only)

Create the initial migration for your existing models:

```bash
# Load environment variables
source .env  # or: export $(cat .env | xargs)

# Create initial migration
make db-migrate
# When prompted, enter: "Initial schema with users and roles tables"
```

This will:

- Scan your `src/db/models.py`
- Generate migration script in `alembic/versions/`
- Detect `DBUser`, `DBRole`, and `user_roles` tables

### Step 3. Apply Migrations and Seed Data

```bash
# Complete setup (migrations + seed data)
make db-init

# Or do it step by step
make db-upgrade  # 1. Apply migrations only
make db-seed     # 2. Add initial data (roles)
```

**What gets seeded:**

- Default roles: `user`, `admin`, `moderator`
- The seed script is idempotent - safe to run multiple times

**Note:** Alembic only creates the table structure (schema). You must run the seed script to populate initial data.

## Daily Workflow

### Adding a New Table or Column

#### Step 1. **Modify your models** in `src/db/models.py`

```python
class DBUser(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True)
    # NEW FIELD:
    phone_number: Mapped[str | None] = mapped_column(String(20), nullable=True)
    # ... rest of fields
```

#### Step 2. **Generate migration**

```bash
make db-migrate
# Enter message: "Add phone_number to users table"
```

#### Step 3. **Review the generated migration** in `alembic/versions/`

```python
def upgrade() -> None:
    """Upgrade database schema."""
    op.add_column('users', sa.Column('phone_number', sa.String(20), nullable=True))

def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_column('users', 'phone_number')
```

#### Step 4. **Apply migration**

```bash
make db-upgrade
```

### Checking Migration Status

```bash
# Show current version
make db-current

# Show full history
make db-history
```

### Rolling Back a Migration

```bash
# Rollback one step
make db-downgrade

# Rollback to specific version
alembic downgrade <revision_id>

# Rollback everything (dangerous!)
make db-reset
```

## Common Commands

| Command | Description |
|---------|-------------|
| `make db-init` | Complete setup: migrations + seed data |
| `make db-migrate` | Create new migration (autogenerate from models) |
| `make db-upgrade` | Apply all pending migrations only |
| `make db-seed` | Seed initial data (roles) - safe to run multiple times |
| `make db-downgrade` | Rollback last migration |
| `make db-current` | Show current database version |
| `make db-history` | Show migration history |
| `make db-reset` | Drop all tables (⚠️ destructive!) |

## Manual Alembic Commands

If you need more control:

```bash
# Create empty migration (manual)
alembic revision -m "Add custom index"

# Upgrade to specific revision
alembic upgrade <revision_id>

# Show SQL without applying
alembic upgrade head --sql

# Stamp database (mark as up-to-date without running)
alembic stamp head
```

## Troubleshooting

### Problem: "Target database is not up to date"

```bash
# Check current version
make db-current

# Apply pending migrations
make db-upgrade
```

### Problem: "Alembic can't find my models"

Make sure:

1. Your models are imported in `src/db/models.py`
2. They inherit from `Base`
3. Environment variables are loaded

### Problem: "Migration conflicts with existing data"

For destructive changes (e.g., removing nullable=True):

1. Create migration in two steps:

   ```bash
   # Step 1: Add new column (nullable)
   make db-migrate  # "Add new_field as nullable"
   make db-upgrade

   # Step 2: Populate data, then make it non-nullable
   make db-migrate  # "Make new_field required"
   make db-upgrade
   ```

### Problem: "Want to reset everything"

```bash
# Drop all API tables and start fresh
make db-reset
make db-upgrade

# Or recreate entire database
docker-compose down -v
docker-compose up -d
make db-init
```

## Database Connection Info

### For SQLTools (VS Code)

**API Database:**

- Host: `localhost`
- Port: `5433`
- Database: `bikerental_api`
- Username: `mlflow`
- Password: `mlflow`

**MLflow Database (read-only):**

- Host: `localhost`
- Port: `5433`
- Database: `mlflow`
- Username: `mlflow`
- Password: `mlflow`

### For Docker Services

Inside Docker network, use:

- Host: `mlflow-db`
- Port: `5432`

## Seeding Initial Data

**Important:** Alembic only manages your database **schema** (table structure). It does NOT populate initial data like default roles.

### Why Separate Schema and Data?

- **Schema migrations** (Alembic): Structure changes (add/remove tables, columns)
- **Data seeding** (setup.py): Initial/default data (roles, default users, config)

### Seeding Workflow

After running migrations, seed your database:

```bash
# Complete setup (migrations + seed)
make db-init

# Or separately
make db-upgrade  # 1. Create tables
make db-seed     # 2. Add initial data
```

### What Gets Seeded?

The `src/api/setup.py` script creates:

- **user** role (Regular user)
- **admin** role (Administrator)
- **moderator** role (Moderator)

### When to Re-run Seeding?

The seed script is **idempotent** - safe to run multiple times. Run it when:

- ✅ First time setup
- ✅ After `docker-compose down -v` (volumes deleted)
- ✅ After database reset (`make db-reset`)
- ✅ Adding new default roles/data

It will only create roles that don't exist, so no duplicates!

### Adding Custom Seed Data

Edit `src/api/setup.py`:

```python
def init_db() -> None:
    """Initialize database with default data."""
    db_pool = get_db_pool()
    Base.metadata.create_all(db_pool.engine)

    with db_pool.get_session() as db:
        # Add your custom seeding logic here
        if not get_role_by_name(db, UserRole.SUPER_ADMIN):
            create_role(db, UserRole.SUPER_ADMIN, "Super Administrator")
```

## Best Practices

1. **Always review autogenerated migrations** before applying
2. **Test migrations in development first**
3. **Keep migrations small and focused** (one logical change per migration)
4. **Write meaningful migration messages**
5. **Never edit applied migrations** (create a new one instead)
6. **Backup production data** before migrations
7. **Use transactions** (Alembic does this by default)
8. **Separate schema from data**: Use Alembic for structure, setup.py for initial data
9. **Run seed script after migrations**: Always run `make db-seed` after `make db-upgrade`
10. **Make seed scripts idempotent**: Check if data exists before inserting
11. **Use centralized configuration**: Leverage `app_settings` for consistent database configuration across the application

## Safety Features

- ✅ **Separate databases** prevent MLflow conflicts
- ✅ **Transactions** rollback on error
- ✅ **Version tracking** shows what's applied
- ✅ **Downgrade support** for rollbacks
- ✅ **SQL preview** before applying

## Integration with FastAPI

Your API automatically uses the correct database:

```python
# src/db/models.py uses app_settings.database_url
# which points to bikerental_api, not mlflow
with get_db_session() as db:
    user = get_user_by_username(db, "alice")
```

MLflow uses its own database connection defined in docker-compose:

```yaml
MLFLOW_DB_URI: postgresql://mlflow:mlflow@mlflow-db:5432/mlflow
```

No conflicts! 🎉

## Example Migration Workflow

```bash
# 1. Start services
docker-compose up -d

# 2. Make changes to models
nano src/db/models.py

# 3. Generate migration
make db-migrate

# 4. Review migration
cat alembic/versions/xxx_your_migration.py

# 5. Apply migration
make db-upgrade

# 6. Seed initial data (if needed)
make db-seed

# 7. Verify
make db-current

# 8. Test your API
curl -X POST http://localhost:8000/api/v1/register \
  -H "Content-Type: application/json" \
  -d '{"username": "test", "email": "test@example.com", "password": "secret123", "full_name": "Test User"}'
```

## References

- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [SQLAlchemy ORM](https://docs.sqlalchemy.org/en/20/orm/)
- [PostgreSQL Multi-Database](https://www.postgresql.org/docs/current/manage-ag-overview.html)
