### 3. `run.sh`
This script handles the Ubuntu 22.04 evaluator constraints: creating the environment, installing system-level dependencies for PyCocoTools/OpenCV, installing Python dependencies, and executing the pipeline.

```bash
#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "Starting NAS-FCOS Automated Pipeline..."

# 1. Install necessary system dependencies (Ubuntu 22.04 compliant)
# Evaluator docker containers often lack these graphics/C++ libraries needed for PyCocoTools and PIL
echo "Updating apt and installing system dependencies..."
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-pip python3-dev build-essential libgl1-mesa-glx libglib2.0-0 wget

# 2. Setup Virtual Environment
echo "Creating Virtual Environment..."
python3 -m venv venv
source venv/bin/activate

# 3. Install Python Dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Execute the Python Pipeline End-to-End
echo "Executing Main Pipeline..."
python run.py

echo "Pipeline complete. Check the 'outputs' folder for results."