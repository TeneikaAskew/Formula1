# 🏎️ F1 Claude Code - Quick Start Guide

## 🚀 First Time Setup (2 minutes)

### Windows:
```powershell
# 1. Clone the project and navigate to it
cd your-f1-project #C:\Users\tenei\Documents\GitHub\Formula1> 

# 2. Run setup
.\build-f1-claude.ps1 setup

# 3. Start Jupyter for F1 notebooks
.\build-f1-claude.ps1 jupyter
```

### Mac/Linux:
```bash
# 1. Clone the project and navigate to it
cd your-f1-project

# 2. Make script executable
chmod +x build-f1-claude.sh

# 3. Run setup
./build-f1-claude.sh setup

# 4. Start Jupyter for F1 notebooks
./build-f1-claude.sh jupyter
```

## 📋 Common Commands

| What you want to do | Windows | Mac/Linux |
|-------------------|---------|-----------|
| View help | `.\build-f1-claude.ps1 help` | `./build-f1-claude.sh help` |
| First time setup | `.\build-f1-claude.ps1 setup` | `./build-f1-claude.sh setup` |
| Start Jupyter notebooks | `.\build-f1-claude.ps1 jupyter` | `./build-f1-claude.sh jupyter` |
| Use Claude CLI | `.\build-f1-claude.ps1 claude` | `./build-f1-claude.sh claude` |
| Start everything | `.\build-f1-claude.ps1 all` | `./build-f1-claude.sh all` |
| Stop everything | `.\build-f1-claude.ps1 stop` | `./build-f1-claude.sh stop` |
| Check status | `.\build-f1-claude.ps1 status` | `./build-f1-claude.sh status` |

## 🎯 What Each Command Does

- **setup**: Builds Docker images (only needed once)
- **jupyter**: Starts Jupyter Lab on http://localhost:8888 with F1 notebooks
- **claude**: Opens interactive Claude Code CLI session
- **all**: Starts both Jupyter and Claude services
- **stop**: Stops all running services
- **clean**: Removes everything (containers, images, data)

## 📊 Available F1 Notebooks

Once Jupyter is running, you can access:

1. **F1_Improved_Models.ipynb** - Base predictive models with proper validation
2. **F1_Constructor_Driver_Evaluation.ipynb** - Driver evaluation and ROI analysis
3. **F1_Betting_Market_Models.ipynb** - Betting market predictions and odds

## ⚠️ Prerequisites

- Docker Desktop installed and running
- Your Anthropic API key (will be prompted during setup)

## 🆘 Troubleshooting

**"Docker is not running"**
- Start Docker Desktop

**"Port 8888 already in use"**
- Stop other Jupyter instances or change the port in docker-compose.yml

**Can't access notebooks**
- Wait 5-10 seconds after starting for services to initialize
- Check logs: `.\build-f1-claude.ps1 logs` or `./build-f1-claude.sh logs`