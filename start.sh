#!/bin/bash

echo "Starting FastAPI..."
python -m api.main &

sleep 5

echo "Starting Streamlit..."
streamlit run app.py --server.port=7860 --server.address=0.0.0.0