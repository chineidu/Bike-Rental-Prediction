# Database Migration Guide with Alembic

This guide explains how to use Alembic for database migrations without breaking your MLflow database.

## Table of Contents
<!-- TOC -->

- [Database Migration Guide with Alembic](#database-migration-guide-with-alembic)
  - [Table of Contents](#table-of-contents)
  - [Architecture Overview](#architecture-overview)
  - [Setup](#setup)
    - [Configure Environment Variables Optional](#configure-environment-variables-optional)
    - [Start Fresh Database](#start-fresh-database)
    - [Initialize Alembic First Time Only](#initialize-alembic-first-time-only)
    - [Apply Migrations and Seed Data](#apply-migrations-and-seed-data)
  - [Daily Workflow](#daily-workflow)
    - [Adding a New Table or Column](#adding-a-new-table-or-column)
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

## Setup

### 0. Configure Environment Variables (Optional)

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

### 1. Start Fresh Database

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

### 2. Initialize Alembic (First Time Only)

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

### 3. Apply Migrations and Seed Data

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

1. **Modify your models** in `src/db/models.py`:

```python
class DBUser(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True)
    # NEW FIELD:
    phone_number: Mapped[str | None] = mapped_column(String(20), nullable=True)
    # ... rest of fields
```

2. **Generate migration**:

```bash
make db-migrate
# Enter message: "Add phone_number to users table"
```

3. **Review the generated migration** in `alembic/versions/`:

```python
def upgrade() -> None:
    """Upgrade database schema."""
    op.add_column('users', sa.Column('phone_number', sa.String(20), nullable=True))

def downgrade() -> None:
    """Downgrade database schema."""
    op.drop_column('users', 'phone_number')
```

4. **Apply migration**:

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
