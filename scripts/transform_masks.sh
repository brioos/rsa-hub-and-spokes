#!/bin/bash

CODE_DIR="/mnt/alex/bids_format/code"
MASK_DIR="/mnt/alex/bids_format/derivatives/masks/masks_bori"
FUNC_DIR="/mnt/alex/bids_format/derivatives/feat/ses-02/first_level/phase_mag"


participants=($(cat "${CODE_DIR}/subjects_ses-02.txt"))
masks=($(ls ${MASK_DIR}/*binary.nii.gz))

for sub in "${participants[@]}"; do
    echo "Processing $sub..."

    OUTPUT_DIR="${MASK_DIR}/${sub}"
    mkdir -p "$OUTPUT_DIR" 

    for mask in "${masks[@]}"; do
        echo "Next mask.. $mask on $sub..."

        applywarp -i "$mask" \
                  -r "${FUNC_DIR}/${sub}/MapRun1.feat/reg/example_func.nii.gz" \
                  -o "${MASK_DIR}/${sub}/$(basename ${mask%.nii.gz})_func.nii.gz" \
                  --postmat="${FUNC_DIR}/${sub}/MapRun1.feat/mask_run1_transf.mat" \
                  --interp=nn

        echo "Done w: ${MASK_DIR}/${sub}/$(basename ${mask%.nii.gz})_func.nii.gz"
    done
done
