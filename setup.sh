#!/bin/bash
set -e

echo "Setting up Entity Resolution System for Yale University Library Catalog"

# Create necessary directories
echo "Creating directory structure..."
#mkdir -p data/input
#mkdir -p data/ground_truth
mkdir -p checkpoints
mkdir -p output
mkdir -p cache/embeddings
mkdir -p cache/queries
mkdir -p logs
mkdir -p profiling

# Check if Python 3.10+ is installed
echo "Checking Python version..."
python_version=$(python --version 2>&1 | awk '{print $2}')
major=$(echo $python_version | cut -d. -f1)
minor=$(echo $python_version | cut -d. -f2)

if [ "$major" -lt 3 ] || [ "$major" -eq 3 -a "$minor" -lt 10 ]; then
    echo "Error: Python 3.10 or higher is required. Found $python_version"
    exit 1
fi

# Check if Docker is installed
echo "Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
echo "Checking Docker Compose installation..."
if ! command -v docker compose &> /dev/null; then
    echo "Error: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create .env file for OpenAI API Key if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    echo "# Add your OpenAI API key here" > .env
    echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env
    echo ".env file created. Please edit it to add your OpenAI API key."
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Make main.py executable
chmod +x main.py

echo "Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit the .env file to add your OpenAI API key"
echo "2. Start Weaviate: docker-compose up -d"
echo "3. Run the pipeline: python main.py --stage all"
echo ""
echo "For development mode with a smaller dataset:"
echo "python main.py --stage all --mode development"
