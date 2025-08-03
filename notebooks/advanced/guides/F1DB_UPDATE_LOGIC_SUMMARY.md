# F1DB Data Loader - Enhanced Update Logic

## Summary of Changes

The `f1db_data_loader.py` has been enhanced to check both version AND race dates when determining if an update is needed.

## New Features

### 1. Race Date Checking (`check_if_update_needed` method)
The loader now checks multiple conditions to determine if an update is needed:

- **Version mismatch**: If current version differs from latest GitHub release
- **Past race detection**: If current date is past a scheduled race date (new results may be available)  
- **Stale data detection**: If data is more than 14 days old
- **Version file age**: If version file is older than 7 days (triggers a check)

### 2. Enhanced Version Command
```bash
python f1db_data_loader.py version
```
Now displays:
- Current and latest versions
- Last update timestamp
- Most recent race in the data
- Next scheduled race
- Update recommendation with specific reason

### 3. Smart Download Command
```bash
python f1db_data_loader.py download
```
- Checks if update is needed before downloading
- Shows why update is/isn't needed
- Use `--force` or `-f` flag to skip checks and force download

### 4. Update Metadata
The loader now saves additional metadata in `.f1db_metadata.json`:
- Version number
- Update timestamp
- Update source
- Data format

## Usage Examples

### Check if data needs updating:
```bash
python f1db_data_loader.py version

# Output:
# Current version: v2025.13.0
# Latest version: v2025.13.0
# Most recent race in data: Formula 1 Belgian Grand Prix 2025 (2025-07-27)
# Next scheduled race: Formula 1 Dutch Grand Prix 2025 (2025-08-31)
# ⚠️ Update recommended: Past next race date (2025-08-31), new results may be available
```

### Smart download (only if needed):
```bash
python f1db_data_loader.py download
```

### Force download regardless of checks:
```bash
python f1db_data_loader.py download --force
```

## How It Works

1. **Version Check**: Compares `.f1db_version` with latest GitHub release
2. **Race Date Analysis**: 
   - Loads `races.csv` to find most recent race date
   - Checks if any scheduled races have passed since last update
   - Ensures data isn't too stale (>14 days old)
3. **Smart Updates**: Only downloads when truly needed, saving bandwidth and time

This ensures you always have the latest race results without unnecessary downloads!