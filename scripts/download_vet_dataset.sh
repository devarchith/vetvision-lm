#!/usr/bin/env bash
# =============================================================================
# Download TUFTS Veterinary Chest X-Ray dataset from Kaggle
#
# Requires Kaggle API credentials:
#   ~/.kaggle/kaggle.json  with  {"username":"...","key":"..."}
#
# Dataset: v7labs/vets-chest-xray-competition
# Usage:
#   bash scripts/download_vet_dataset.sh [TARGET_DIR]
#   bash scripts/download_vet_dataset.sh data
# =============================================================================

set -euo pipefail

TARGET_DIR="${1:-data}"
KAGGLE_DATASET="v7labs/vets-chest-xray-competition"
DATASET_DIR="${TARGET_DIR}/vet_dataset"

echo ""
echo "============================================================"
echo "  TUFTS Veterinary Chest X-Ray Dataset Download"
echo "============================================================"

# Check kaggle CLI
if ! command -v kaggle &> /dev/null; then
    echo "[ERROR] kaggle CLI not found. Install with:"
    echo "  pip install kaggle"
    echo "  mkdir -p ~/.kaggle"
    echo "  cp your_kaggle.json ~/.kaggle/kaggle.json"
    echo "  chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

# Check credentials
if [ ! -f "${HOME}/.kaggle/kaggle.json" ]; then
    echo "[ERROR] Kaggle API credentials not found at ~/.kaggle/kaggle.json"
    echo ""
    echo "To set up credentials:"
    echo "  1. Log in to https://www.kaggle.com"
    echo "  2. Go to Account → API → Create New API Token"
    echo "  3. Place the downloaded kaggle.json at ~/.kaggle/kaggle.json"
    echo "  4. chmod 600 ~/.kaggle/kaggle.json"
    exit 1
fi

echo "[INFO] Downloading dataset: ${KAGGLE_DATASET}"
mkdir -p "${DATASET_DIR}"

kaggle datasets download \
    -d "${KAGGLE_DATASET}" \
    -p "${DATASET_DIR}" \
    --unzip

echo ""
echo "[INFO] Download complete. Contents:"
ls -lh "${DATASET_DIR}" | head -20

echo ""
echo "[NEXT] Generate the manifest CSV:"
echo "  python scripts/generate_manifest.py --data-dir ${DATASET_DIR} --output ${DATASET_DIR}/manifest.csv"
echo ""
echo "[NEXT] Update configs/finetune.yaml:"
echo "  data:"
echo "    vet_root:     \"${DATASET_DIR}\""
echo "    manifest_csv: \"${DATASET_DIR}/manifest.csv\""
echo "============================================================"
