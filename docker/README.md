# Docker Configuration for F1 Claude Code

This folder contains all Docker-related files for the F1 predictive solution project.

## 📁 Files Overview

### Dockerfiles

1. **Dockerfile.jupyter** (Main)
   - Primary Docker image for running Jupyter notebooks
   - Python 3.10 base with ML packages
   - Includes automatic F1 data extraction
   - Ports: 8888 (Jupyter), 5000 (API)
   - **Use this for:** Running the F1 analysis notebooks

2. **Dockerfile.claude-code**
   - Minimal Claude Code CLI setup
   - Node.js 18 base with Claude Code npm package
   - Basic Python support
   - **Use this for:** Claude Code CLI only

3. **Dockerfile.dev** (Development)
   - Feature-rich development environment
   - Includes Streamlit, FastAPI, Plotly, Dash
   - Non-root user setup for security
   - Multiple ports exposed
   - **Use this for:** Advanced development with multiple frameworks

### docker-compose.yml

Defines three services:

- **claude-code**: Claude Code CLI service
- **f1-jupyter**: Main Jupyter service (default)
- **f1-dev**: Development environment (profile: dev)

## 🚀 Usage

All Docker operations should be run from the project root using the helper scripts:

**Windows:**
```powershell
.\build-f1-claude.ps1 [command]
```

**Mac/Linux:**
```bash
./build-f1-claude.sh [command]
```

## 🏗️ Architecture Decisions

- **Separated Dockerfiles**: Each serves a specific purpose rather than one bloated image
- **Context set to parent**: All builds use `context: ..` to access project files
- **Named volumes**: F1 data persists in `f1-data` volume
- **Profiles**: Development environment only starts when explicitly requested
- **Network isolation**: All services use `f1-network` for communication