# F1 Claude Code - Easy Manager Script for Windows
# Usage: .\build-f1-claude.ps1 [command]

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Info { Write-Host $args -ForegroundColor Cyan }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }

# Check if Docker is running
function Test-Docker {
    try {
        docker ps 2>&1 | Out-Null
        return $true
    } catch {
        Write-Error "Docker is not running. Please start Docker Desktop."
        return $false
    }
}

# Check if .env file exists
function Test-EnvFile {
    if (-not (Test-Path ".env")) {
        Write-Warning ".env file not found. Creating from template..."
        if (Test-Path ".env.example") {
            Copy-Item ".env.example" ".env"
            Write-Warning "Please edit .env and add your ANTHROPIC_API_KEY"
            notepad .env
            Read-Host "Press Enter after adding your API key"
        } else {
            Write-Error ".env.example not found!"
            return $false
        }
    }
    return $true
}

# Main script
Write-Host ""
Write-Host "🏎️  F1 Claude Code Manager" -ForegroundColor Green
Write-Host "=========================" -ForegroundColor Green
Write-Host ""

if (-not (Test-Docker)) { exit 1 }

switch ($Command.ToLower()) {
    "help" {
        Write-Info "Available commands:"
        Write-Host "  .\build-f1-claude.ps1 setup      - First time setup (build images)"
        Write-Host "  .\build-f1-claude.ps1 jupyter    - Start Jupyter Lab for F1 notebooks"
        Write-Host "  .\build-f1-claude.ps1 claude     - Start Claude Code CLI"
        Write-Host "  .\build-f1-claude.ps1 all        - Start all services"
        Write-Host "  .\build-f1-claude.ps1 stop       - Stop all services"
        Write-Host "  .\build-f1-claude.ps1 clean      - Stop and remove all containers/volumes"
        Write-Host "  .\build-f1-claude.ps1 status     - Show running services"
        Write-Host "  .\build-f1-claude.ps1 logs       - Show logs"
        Write-Host ""
    }
    
    "setup" {
        Write-Info "🔧 Setting up F1 Claude Code..."
        if (-not (Test-EnvFile)) { exit 1 }
        
        Write-Info "Building Docker images..."
        docker-compose -f docker/docker-compose.yml -f docker/docker-compose -f docker/docker-compose.yml.yml build
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "`n✅ Setup complete! Run '.\build-f1-claude.ps1 jupyter' to start."
        } else {
            Write-Error "Build failed!"
            exit 1
        }
    }
    
    "jupyter" {
        Write-Info "🚀 Starting Jupyter Lab..."
        if (-not (Test-EnvFile)) { exit 1 }
        
        docker-compose -f docker/docker-compose.yml up -d f1-jupyter
        
        if ($LASTEXITCODE -eq 0) {
            Start-Sleep -Seconds 3
            Write-Success "`n✅ Jupyter Lab is running!"
            Write-Info "📊 Open http://localhost:8888 in your browser"
            Write-Info "📝 Available notebooks:"
            Write-Host "   - F1_Core_Models.ipynb"
            Write-Host "   - F1_Constructor_Driver_Evaluation.ipynb"
            Write-Host "   - F1_Betting_Market_Models.ipynb"
            Write-Host ""
            Write-Info "To stop: .\build-f1-claude.ps1 stop"
            
            # Optionally open browser
            $openBrowser = Read-Host "Open browser now? (y/n)"
            if ($openBrowser -eq "y") {
                Start-Process "http://localhost:8888"
            }
        }
    }
    
    "claude" {
        Write-Info "🤖 Starting Claude Code CLI..."
        if (-not (Test-EnvFile)) { exit 1 }
        
        Write-Info "Entering interactive Claude session (type 'exit' to quit)..."
        docker-compose -f docker/docker-compose.yml run --rm claude-code claude
    }
    
    "all" {
        Write-Info "🚀 Starting all services..."
        if (-not (Test-EnvFile)) { exit 1 }
        
        docker-compose -f docker/docker-compose.yml up -d
        
        if ($LASTEXITCODE -eq 0) {
            Start-Sleep -Seconds 3
            Write-Success "`n✅ All services started!"
            Write-Info "📊 Jupyter Lab: http://localhost:8888"
            Write-Info "🤖 Claude CLI: docker exec -it claude-code-cli bash"
            Write-Host ""
            Write-Info "Quick commands:"
            Write-Host "  Claude: docker exec -it claude-code-cli claude"
            Write-Host "  Bash:   docker exec -it claude-code-cli bash"
        }
    }
    
    "stop" {
        Write-Info "🛑 Stopping all services..."
        docker-compose -f docker/docker-compose.yml down
        Write-Success "✅ All services stopped."
    }
    
    "clean" {
        Write-Warning "⚠️  This will remove all containers, images, and data!"
        $confirm = Read-Host "Are you sure? (yes/no)"
        if ($confirm -eq "yes") {
            Write-Info "🧹 Cleaning up..."
            docker-compose -f docker/docker-compose.yml down -v --rmi all
            Write-Success "✅ Cleanup complete."
        } else {
            Write-Info "Cancelled."
        }
    }
    
    "status" {
        Write-Info "📊 Service Status:"
        docker-compose -f docker/docker-compose.yml ps
    }
    
    "logs" {
        Write-Info "📜 Showing logs (Ctrl+C to exit)..."
        docker-compose -f docker/docker-compose.yml logs -f
    }
    
    default {
        Write-Error "Unknown command: $Command"
        Write-Host "Run '.\build-f1-claude.ps1 help' for available commands."
        exit 1
    }
}
