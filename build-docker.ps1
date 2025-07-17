Write-Host "Formula 1 Machine Learning Workshop - Docker Setup" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green

# Build the Docker image
Write-Host "Building Docker image..." -ForegroundColor Yellow
docker build -t formula1-ml-workshop .

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Docker image built successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "To run the container:" -ForegroundColor Cyan
    Write-Host "  docker run -p 8888:8888 -v ${PWD}:/app formula1-ml-workshop" -ForegroundColor White
    Write-Host ""
    Write-Host "Or use docker-compose:" -ForegroundColor Cyan
    Write-Host "  docker-compose up" -ForegroundColor White
    Write-Host ""
    Write-Host "Then open your browser to: http://localhost:8888" -ForegroundColor Yellow
} else {
    Write-Host "Docker build failed!" -ForegroundColor Red
}
