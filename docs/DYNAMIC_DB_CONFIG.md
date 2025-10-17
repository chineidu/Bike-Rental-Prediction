# Dynamic Database Configuration Summary

## What Changed

The database initialization script now uses environment variables instead of hardcoded values.

### Files Modified

1. **`docker/init-databases.sh`**
   - Now reads `API_DB_NAME` from environment
   - Falls back to `bikerental_api` if not set
   - Uses `${POSTGRES_USER}` and `${POSTGRES_DB}` from environment

2. **`docker-compose.yaml`**
   - Added `API_DB_NAME` to `mlflow-db` environment variables
   - Passes the value to the init script

3. **`docs/ALEMBIC_GUIDE.md`**
   - Updated to show dynamic configuration
   - Added environment variable configuration section

4. **`docs/ENV_VARS.md`** (NEW)
   - Complete reference for all environment variables
   - Examples and troubleshooting

## Benefits

✅ **Flexible Configuration**

- Change database names without editing scripts
- Environment-specific configurations (dev, staging, prod)

✅ **Consistent Behavior**

- Same variable names across all files
- Defaults prevent breaking changes

✅ **Easy Customization**

- Just edit `.env` file
- No code changes required

## Usage

### Using Default Names

No changes needed! Just run:

```bash
docker-compose up -d
```

Creates:

- `mlflow` database (for MLflow)
- `bikerental_api` database (for API)

### Using Custom Names

Edit `.env`:

```bash
# Change MLflow database name
POSTGRES_DB=my_mlflow_tracking

# Change API database name
API_DB_NAME=my_api_database
```

Then recreate:

```bash
docker-compose down -v
docker-compose up -d
```

Creates:

- `my_mlflow_tracking` database
- `my_api_database` database

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `POSTGRES_DB` | MLflow database name | `mlflow` |
| `POSTGRES_USER` | PostgreSQL username | `mlflow` |
| `POSTGRES_PASSWORD` | PostgreSQL password | `mlflow` |
| `API_DB_NAME` | API database name | `bikerental_api` |

## Verification

Check that databases were created:

```bash
# Connect to PostgreSQL
docker-compose exec mlflow-db psql -U mlflow

# List databases
\l

# Expected output:
#                                   List of databases
#       Name       |  Owner   | Encoding |  Collate   |   Ctype    |
# -----------------+----------+----------+------------+------------+
#  bikerental_api  | mlflow   | UTF8     | en_US.utf8 | en_US.utf8 |
#  mlflow          | mlflow   | UTF8     | en_US.utf8 | en_US.utf8 |
#  postgres        | mlflow   | UTF8     | en_US.utf8 | en_US.utf8 |
```

## Migration Path

If you have existing data:

1. **Backup existing data:**

   ```bash
   docker-compose exec mlflow-db pg_dump -U mlflow mlflow > mlflow_backup.sql
   docker-compose exec mlflow-db pg_dump -U mlflow bikerental_api > api_backup.sql
   ```

2. **Change names in `.env`:**

   ```bash
   POSTGRES_DB=new_mlflow_name
   API_DB_NAME=new_api_name
   ```

3. **Recreate with new names:**

   ```bash
   docker-compose down -v
   docker-compose up -d
   ```

4. **Restore data:**

   ```bash
   docker-compose exec -T mlflow-db psql -U mlflow new_mlflow_name < mlflow_backup.sql
   docker-compose exec -T mlflow-db psql -U mlflow new_api_name < api_backup.sql
   ```

## Testing

```bash
# 1. Test with default names
docker-compose down -v
docker-compose up -d
docker-compose exec mlflow-db psql -U mlflow -c "\l"

# 2. Test with custom names
echo "API_DB_NAME=test_api_db" >> .env
docker-compose down -v
docker-compose up -d
docker-compose exec mlflow-db psql -U mlflow -c "\l" | grep test_api_db
```

## Notes

- ⚠️ Database names must be valid PostgreSQL identifiers (no spaces, special chars)
- ✅ Init script only runs on first container startup (not on restarts)
- ✅ Changes require recreating volumes (`docker-compose down -v`)
- ✅ Alembic automatically uses the configured API database name
