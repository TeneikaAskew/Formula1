# F1 Claude Code - Docker Usage Guide

## Overview

This project includes two Docker setups:
1. **Claude Code CLI** - For running Claude Code commands
2. **Jupyter Services** - For running F1 analysis notebooks

## Quick Start

### 1. Set up Environment

```bash
# Create .env file with your Anthropic API key
echo "ANTHROPIC_API_KEY=your-api-key-here" > .env
```

### 2. Build and Run Services

```bash
# Build all services
docker-compose -f docker/docker-compose.yml build

# Start Jupyter service (for notebooks)
docker-compose -f docker/docker-compose.yml up f1-jupyter -d

# Access Claude Code CLI
docker-compose -f docker/docker-compose.yml run --rm claude-code claude

# Or use the helper scripts (recommended):
# Windows: .\build-f1-claude.ps1 setup
# Mac/Linux: ./build-f1-claude.sh setup
```

Access JupyterLab at: http://localhost:8888

## Claude Code CLI Usage

### Running Claude Code Commands

```bash
# Interactive Claude session
docker-compose run --rm claude-code claude

# Run Claude with a specific prompt
docker-compose run --rm claude-code claude "analyze the F1 notebooks"

# Access bash in Claude Code container
docker-compose run --rm claude-code bash

# Keep Claude Code container running and exec into it
docker-compose up -d claude-code
docker exec -it claude-code-cli claude
```

### Claude Code Configuration

The Claude Code configuration is persisted in `~/.claude-code` on your host machine.

```bash
# Configure Claude Code (first time setup)
docker-compose run --rm claude-code claude configure

# Check Claude Code version
docker-compose run --rm claude-code claude --version
```

## Jupyter Notebook Usage

### Build and Run with Docker Compose

```bash
# Build the image
docker build -t f1-claude-code .

# Run JupyterLab
docker run -p 8888:8888 -v $(pwd):/app f1-claude-code

# Run with interactive bash
docker run -it -p 8888:8888 -v $(pwd):/app f1-claude-code bash
```

## Available Services

### Main Service: f1-claude-code
- **Port 8888**: JupyterLab interface
- **Port 5000**: Reserved for future API endpoints
- **Default Command**: Starts JupyterLab

### Alternative Service: f1-notebook
- **Port 8889**: Classic Jupyter Notebook
- **Profile**: notebook (requires explicit activation)

To run the classic notebook interface:
```bash
docker-compose --profile notebook up
```

## Docker Commands

### Start Services
```bash
# Start main JupyterLab service
docker-compose up

# Start both JupyterLab and classic notebook
docker-compose --profile notebook up

# Run in background
docker-compose up -d
```

### Stop Services
```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: removes data)
docker-compose down -v
```

### Access Running Container
```bash
# Execute bash in running container
docker exec -it f1-claude-code bash

# Run Python commands
docker exec -it f1-claude-code python -c "print('Hello F1')"
```

### View Logs
```bash
# View all logs
docker-compose logs

# Follow logs in real-time
docker-compose logs -f

# View specific service logs
docker-compose logs f1-claude-code
```

## Volume Management

The setup uses Docker volumes to persist data:

- **f1-data**: Stores extracted CSV files persistently
- **./**: Mounted to /app for live code updates

### Data Persistence
```bash
# List volumes
docker volume ls

# Inspect data volume
docker volume inspect f1-claude-code_f1-data

# Backup data volume
docker run --rm -v f1-claude-code_f1-data:/data -v $(pwd):/backup alpine tar czf /backup/f1-data-backup.tar.gz -C /data .
```

## Different Startup Modes

The Docker image supports multiple startup modes:

```bash
# JupyterLab (default)
docker run -p 8888:8888 f1-claude-code jupyter

# Classic Jupyter Notebook
docker run -p 8888:8888 f1-claude-code notebook

# Interactive bash shell
docker run -it f1-claude-code bash

# Run specific Python script
docker run -v $(pwd):/app f1-claude-code python your_script.py
```

## Environment Variables

You can customize the behavior using environment variables:

```bash
# Disable JupyterLab (use classic notebook)
docker run -e JUPYTER_ENABLE_LAB=no -p 8888:8888 f1-claude-code

# Custom Jupyter settings
docker run -e JUPYTER_CONFIG_DIR=/app/config -p 8888:8888 f1-claude-code
```

## Troubleshooting

### Port Already in Use
```bash
# Check what's using port 8888
lsof -i :8888

# Use a different port
docker run -p 8889:8888 f1-claude-code
```

### Data Extraction Issues
```bash
# Manually extract data in container
docker exec -it f1-claude-code bash
unzip -q f1db_csv.zip -d data/
```

### Permission Issues
```bash
# Run with root privileges (if needed)
docker run --user root -p 8888:8888 f1-claude-code
```

### Clean Everything
```bash
# Stop all containers and remove everything
docker-compose down -v --rmi all
docker system prune -a
```

## Production Considerations

For production deployment:

1. **Add Authentication**: Remove `--NotebookApp.token=""` and set a secure token
2. **Use HTTPS**: Add SSL certificates and reverse proxy
3. **Resource Limits**: Add resource constraints in docker-compose.yml
4. **Monitoring**: Add health checks and logging

Example production config:
```yaml
services:
  f1-claude-code:
    # ... other settings ...
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8888/api"]
      interval: 30s
      timeout: 10s
      retries: 3
```

## Accessing the F1 Notebooks

Once JupyterLab is running:

1. Open http://localhost:8888 in your browser
2. Navigate to the notebooks:
   - `F1_Improved_Models.ipynb` - Base predictive models
   - `F1_Constructor_Driver_Evaluation.ipynb` - Driver evaluation system
   - `F1_Betting_Market_Models.ipynb` - Betting market predictions

The notebooks will automatically extract the F1 data on first run if needed.