#!/bin/bash

# Activate the virtual environment
# Using the path observed in tel_sslm.sh
source ~/sslm-venv/bin/activate

# Ensure dependencies
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "Streamlit not found. Installing..."
    pip3 install streamlit
fi

if ! python3 -c "import huggingface_hub" &> /dev/null; then
    echo "huggingface_hub not found. Installing..."
    pip3 install huggingface_hub
fi

# Run the app
echo "Starting SSLM Visualizer..."
# Assuming we are running from the parent directory or sslm-visualization directory
# If run from parent (transformer-sslm):
if [ -f "sslm-visualization/app.py" ]; then
    streamlit run sslm-visualization/app.py --server.port 8501 --server.address 0.0.0.0
elif [ -f "app.py" ]; then
    streamlit run app.py --server.port 8501 --server.address 0.0.0.0
else
    echo "Error: app.py not found!"
    exit 1
fi
