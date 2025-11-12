#!/bin/bash
# Setup script for the benchmarking framework

echo "Setting up LLM Metadata Curation Benchmarking Framework..."

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Set AWS credentials:"
echo "   export AWS_ACCESS_KEY_ID=your_key"
echo "   export AWS_SECRET_ACCESS_KEY=your_secret"
echo "3. List available tasks: python -m src.cli list"
echo "4. Run an experiment: python -m src.cli run <task_name>"

