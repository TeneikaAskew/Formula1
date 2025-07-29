#!/bin/bash
# Load environment variables from .env file

# Find the .env file
ENV_FILE="../../.env"

if [ -f "$ENV_FILE" ]; then
    echo "Loading environment variables from $ENV_FILE"
    
    # Export each non-comment line
    while IFS='=' read -r key value; do
        # Skip comments and empty lines
        if [[ ! "$key" =~ ^[[:space:]]*# ]] && [[ -n "$key" ]]; then
            # Remove any quotes from the value
            value="${value%\"}"
            value="${value#\"}"
            export "$key=$value"
            echo "Loaded: $key"
        fi
    done < "$ENV_FILE"
    
    echo "Environment variables loaded successfully!"
else
    echo "Error: .env file not found at $ENV_FILE"
    exit 1
fi