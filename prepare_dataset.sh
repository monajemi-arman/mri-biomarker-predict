#!/bin/bash

SRC_DIR="PKG - UCSF-PDGM Version 5"
IMG_DIR="dataset/images"
MASK_DIR="dataset/masks"

mkdir -p "$IMG_DIR"
mkdir -p "$MASK_DIR"

find "$SRC_DIR" -type f -name "*.nii.gz" | while read -r file; do
    filename=$(basename "$file")

    if [[ "$filename" == *"segmentation"* ]]; then
        mv "$file" "$MASK_DIR/"
        echo "MASK  -> $filename"
    else
        mv "$file" "$IMG_DIR/"
        echo "IMAGE -> $filename"
    fi
done