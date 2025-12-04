#!/bin/bash
echo "-----------------------------------------"
echo "      Starting OpenVoice API Server       "
echo "-----------------------------------------"

LOGFILE="/workspace/openvoice-api/server.log"
ENV_NAME="openvoice"
APP_PATH="/workspace/openvoice-api/server.py"
PORT=8000

echo "[INFO] Activating conda environment: $ENV_NAME"
source /workspace/miniconda/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "[INFO] Installing required NLTK models (first run only)..."

python3 << 'EOF'
import nltk

resources = [
    "taggers/averaged_perceptron_tagger_eng",
    "tokenizers/punkt",
    "corpora/wordnet"
]

for res in resources:
    try:
        nltk.data.find(res)
        print(f"[NLTK] Already installed: {res}")
    except LookupError:
        print(f"[NLTK] Downloading: {res}")
        nltk.download(res.split("/")[-1])
EOF

echo "[INFO] Checking for existing OpenVoice server..."
PID=$(pgrep -f "uvicorn.*server:app")
if [[ ! -z "$PID" ]]; then
  echo "[INFO] Killing existing server process: $PID"
  kill -9 $PID
fi

echo "[INFO] Launching OpenVoice FastAPI server on port $PORT..."
nohup uvicorn server:app \
  --host 0.0.0.0 \
  --port $PORT \
  --workers 1 \
  >> $LOGFILE 2>&1 &

sleep 1

NEW_PID=$(pgrep -f "uvicorn.*server:app")
echo "[INFO] Server started with PID: $NEW_PID"
echo "[INFO] Logs at: $LOGFILE"
echo "-----------------------------------------"
echo " OpenVoice API is now running 🎤"
echo "-----------------------------------------"