#!/bin/bash
MASK_DIR="/mnt/c/Documents/Programozás/rsa-hub-and-spokes/masks_bori"

masks=($(ls ${MASK_DIR}/*.nii.gz))
for file in "${masks[@]}"; do
    output="${file%.nii.gz}_binary.nii.gz"
    fslmaths "$file" -thr 25 -bin "$output"
    
    echo "Processed $file -> $output"
done
