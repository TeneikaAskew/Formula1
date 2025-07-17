#!/bin/bash

echo "Formula 1 Machine Learning Workshop - Docker Setup"
echo "=================================================="

# Build the Docker image
echo "Building Docker image..."
docker build -t formula1-ml-workshop .

echo ""
echo "Docker image built successfully!"
echo ""
echo "To run the container:"
echo "  docker run -p 8888:8888 -v \$(pwd):/app formula1-ml-workshop"
echo ""
echo "Or use docker-compose:"
echo "  docker-compose up"
echo ""
echo "Then open your browser to: http://localhost:8888"
