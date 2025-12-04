#!/bin/bash
# Download Amazon Reviews 2023 dataset from Hugging Face

echo "Creating data directory..."
mkdir -p data

echo ""
echo "=========================================="
echo "Downloading All_Beauty reviews (~327 MB)"
echo "=========================================="
curl -L "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/review_categories/All_Beauty.jsonl" \
     -o data/All_Beauty.jsonl \
     --progress-bar

echo ""
echo "=========================================="
echo "Downloading All_Beauty metadata (~61 MB)"
echo "=========================================="
curl -L "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/raw/meta_categories/meta_All_Beauty.jsonl" \
     -o data/meta_All_Beauty.jsonl \
     --progress-bar

echo ""
echo "=========================================="
echo "Download complete!"
echo "=========================================="
echo ""
echo "Files saved to data/"
ls -lh data/*.jsonl

echo ""
echo "Compressing files to save space..."
gzip -f data/All_Beauty.jsonl
gzip -f data/meta_All_Beauty.jsonl

echo ""
echo "Final files:"
ls -lh data/*.jsonl.gz
echo ""
echo "Ready to use with Data_v2.ipynb!"
