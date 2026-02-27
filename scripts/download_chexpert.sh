#!/usr/bin/env bash
# =============================================================================
# Download CheXpert dataset (Stanford AIMI)
#
# CheXpert requires acceptance of a data-use agreement before download.
# This script prints the required steps and verifies the expected layout.
#
# Usage:
#   bash scripts/download_chexpert.sh [TARGET_DIR]
#   bash scripts/download_chexpert.sh data
# =============================================================================

set -euo pipefail

TARGET_DIR="${1:-data}"
DATASET_URL="https://stanfordaimi.azurewebsites.net/datasets/8cbd9ed4-2eb9-4565-affc-111cf4f7ebe2"

echo ""
echo "============================================================"
echo "  CheXpert Dataset Download"
echo "============================================================"
echo ""
echo "CheXpert is gated behind a data-use agreement."
echo ""
echo "STEP 1 — Request access:"
echo "  Visit: ${DATASET_URL}"
echo "  Complete the form (academic / non-commercial use)."
echo ""
echo "STEP 2 — Download the small version (~11 GB):"
echo "  File: CheXpert-v1.0-small.zip"
echo ""
echo "STEP 3 — Extract to the target directory:"
echo "  mkdir -p ${TARGET_DIR}"
echo "  unzip CheXpert-v1.0-small.zip -d ${TARGET_DIR}/"
echo ""
echo "STEP 4 — Expected layout after extraction:"
echo "  ${TARGET_DIR}/CheXpert-v1.0-small/"
echo "    train.csv"
echo "    valid.csv"
echo "    train/"
echo "      patient00001/"
echo "        study1/"
echo "          view1_frontal.jpg"
echo ""
echo "STEP 5 — Update configs/pretrain.yaml:"
echo "  data:"
echo "    chexpert_root: \"${TARGET_DIR}/CheXpert-v1.0-small\""
echo "    train_csv:     \"${TARGET_DIR}/CheXpert-v1.0-small/train.csv\""
echo "    valid_csv:     \"${TARGET_DIR}/CheXpert-v1.0-small/valid.csv\""
echo ""
echo "============================================================"

# Optionally verify an existing download
CHEXPERT_DIR="${TARGET_DIR}/CheXpert-v1.0-small"
if [ -d "${CHEXPERT_DIR}" ]; then
    echo ""
    echo "[CHECK] Found existing directory: ${CHEXPERT_DIR}"
    for f in train.csv valid.csv; do
        if [ -f "${CHEXPERT_DIR}/${f}" ]; then
            echo "  [OK] ${f}"
        else
            echo "  [MISSING] ${f}"
        fi
    done
else
    echo ""
    echo "[INFO] Dataset not yet downloaded. Follow the steps above."
fi
