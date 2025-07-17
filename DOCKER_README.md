# Formula 1 Docker Setup

Your Formula 1 Machine Learning Workshop is now dockerized! 🏎️

## Files Created

- `Dockerfile` - Docker configuration for your Python environment
- `requirements.txt` - Python dependencies 
- `docker-compose.yml` - Easy container management
- `.dockerignore` - Files to exclude from Docker build
- `build-docker.ps1` - PowerShell build script
- `build-docker.sh` - Bash build script

## How to Run

### Option 1: Using Docker Compose (Recommended)
```powershell
docker-compose up
```

### Option 2: Using Docker Run
```powershell
docker run -p 8888:8888 -v ${PWD}:/app formula1-ml-workshop
```

### Option 3: Using the PowerShell Script
```powershell
.\build-docker.ps1
```

## Access Your Notebooks

After running the container, open your browser to:
**http://localhost:8888**

You'll see the Jupyter notebook interface with access to all your Formula 1 notebooks and data.

## What's Included

- Python 3.10
- Jupyter Notebook
- All required packages (pandas, matplotlib, seaborn, scikit-learn, numpy)
- Your F1 data and notebooks
- Pre-configured environment

## Stopping the Container

Press `Ctrl+C` in the terminal or run:
```powershell
docker-compose down
```

## Rebuilding (if you make changes)

```powershell
docker-compose build --no-cache
```

## Troubleshooting

1. **Port already in use**: Change the port in `docker-compose.yml` from `8888:8888` to `8889:8888`
2. **Permission issues**: Make sure Docker Desktop is running
3. **Build fails**: Try running `docker system prune` to clean up Docker cache

Happy racing! 🏁
