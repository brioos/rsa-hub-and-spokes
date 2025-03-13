# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 20:32:05 2025

@author: borib
"""

#!/usr/bin/env python
# coding: utf-8

import os
import os.path as op
import time
import gc
import numpy as np
import scipy
import scipy.sparse
import nibabel as nib
import pandas as pd
import matplotlib.pyplot as plt
from pprint import pprint
from tqdm import tqdm
import warnings
from slicetime.main import run_slicetime
from glmsingle.glmsingle import GLM_single

warnings.filterwarnings('ignore')

# %%

## Set dir
mri_folder = "/mnt/alex/bids_format/derivatives/feat/ses-02/first_level/phase_mag"
onsets_folder = "/mnt/alex/bids_format/derivatives/onsets/ses-02_upsampled/"
glm_single_folder = "/mnt/alex/bids_format/derivatives/upsample/ses-02/glmsingle/"
if not os.path.exists(glm_single_folder):
    os.makedirs(glm_single_folder, exist_ok=True)
out_folder = "/mnt/alex/bids_format/derivatives/upsample/ses-02/"
code_folder = "/mnt/alex/bids_format/code/"


## Parameters
stimdur = 2
tr_old = 1.15
tr_new = 1
n_slices = 80
RunNum = 8
sumTR = 654


participant_list = open(os.path.join(code_folder, "subjects_ses-02.txt")).read().splitlines()

## ID corresp
expid_mrid_map = {}
with open(os.path.join(code_folder, "code_pairs_ses-02.txt"), "r") as file:
    next(file)
    for line in file:
        expid, participant_id = line.strip().split("\t")
        expid_mrid_map[expid] = participant_id


## Upsampling
def Upsample(participant_list):
    for ParticipantID in participant_list:
        for r in ['1', '2']:
            r = int(r)
            if r == 1:
                run_group = [1, 2, 3, 4]
            else:
                run_group = [5, 6, 7, 8]
            print(f"Processing Subject: {ParticipantID}, Group {r} (Runs: {run_group})")

            for run_num in run_group:
                start_time = time.time()
            
                original_fname = f"{ParticipantID}/MapRun{run_num}.feat/filtered_func_data_EF1reg.nii.gz"
                upsampled_fname = f"{ParticipantID}/MapRun{run_num}.feat/upsampled_filtered_func_data_EF1reg.nii.gz"
                in_file = os.path.join(mri_folder, original_fname)
                out_file = os.path.join(out_folder, upsampled_fname)
                out_dir = os.path.join(out_folder, f"{ParticipantID}/MapRun{run_num}.feat/")

                if not os.path.exists(in_file):
                    print(f"WARNING: Input file not found: {in_file}. Skipping...")
                    continue
                
                if os.path.exists(out_file):
                    print(f"✅ Output file already exists: {out_file}. Skipping...")
                    continue
                
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)
            
                slicetimes = np.flip(np.arange(0, tr_old, tr_old / n_slices))
                run_slicetime(inpath=in_file, outpath=out_file, slicetimes=slicetimes, tr_old=tr_old, tr_new=tr_new)
            
                print(f"Upsampled: {out_file}")
                print(f"Elapsed time: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}\n")
                oimg = nib.load(in_file)
                print(oimg.shape)
                print(oimg.header)
                img = nib.load(out_file)
                print(img.shape)
                print(img.header)
                plt.subplot(211)
                plt.plot(oimg.get_fdata()[75,75,35,:])
                plt.subplot(212)
                plt.plot(img.get_fdata()[75,75,35,:])
                plt.savefig(os.path.join(out_folder, f"{ParticipantID}/MapRun{run_num}.feat/upsampled_diff.png"))
## GLMSingle
def GLMsingleWrapper(expid, ParticipantID, inputf):
    glm_single_folder_participant = os.path.join(glm_single_folder, ParticipantID, inputf.split('.')[0])
    os.makedirs(glm_single_folder_participant, exist_ok=True)
    
    output_file = os.path.join(glm_single_folder_participant, "TYPED_FITHRF_GLMDENOISE_RR.npy")

    if os.path.exists(output_file):
        print("✅ GLMsingle output already exists. Skipping...")
        return
    
    participant_fmri_data = CollectFMRI(ParticipantID, inputf)
    participant_csr_designs = CollectConds(ParticipantID, expid)
    RunGLMsingle(participant_csr_designs, participant_fmri_data, inputf, glm_single_folder_participant)
# %%

def CollectConds(ParticipantID, expid):
    participant_onset_folder = os.path.join(onsets_folder, expid)
    participant_stimuli_conds = []
    participant_csr_designs = []

    for run_onset in range(RunNum):
        onset_file = expid + "_Run" + str(run_onset+1) + "_onsets.txt"
        onset_fname = os.path.join(participant_onset_folder, onset_file)
        print("Reading onset file for stimlist: ", onset_fname)

        with open(onset_fname, 'r') as fh: 
            for line in fh:
                participant_stimuli_conds.append([line.split(" ")[0][0], int(line.split(" ")[0][1:])])
                

    participant_stimuli_conds = [list(x) for x in set(tuple(x) for x in participant_stimuli_conds)]
    participant_stimuli_conds = sorted(participant_stimuli_conds, key=lambda x: (x[1], x[0]))
    participant_stimuli_conds = [str(x[0]) + str(x[1]) for x in participant_stimuli_conds]

    for run_onset in range(RunNum):
        onset_file = expid + "_Run" + str(run_onset+1) + "_onsets.txt"
        onset_fname = os.path.join(participant_onset_folder, onset_file)
        print("Reading onset file for matrix: ", onset_fname)
        run_row, run_col, run_data = [], [], []

        with open(onset_fname, 'r') as fh: 
            for line in fh:
                run_row.append(int(line.split(" ")[1]))
                run_col.append(participant_stimuli_conds.index(line.split(" ")[0]))
                run_data.append(1)
        run_csr = scipy.sparse.csr_matrix((run_data, (run_row, run_col)), shape=(sumTR, len(participant_stimuli_conds))).toarray()
        participant_csr_designs.append(run_csr)
    
    print(f"Participant {ParticipantID}, Experimental ID {expid}, Run {run_onset+1}: Design matrix shape = {run_csr.shape}")
    return participant_csr_designs

def CollectFMRI(ParticipantID, inputf):
    participant_fmri_data = []
    
    participant_fmri_folder = os.path.join(out_folder, ParticipantID)
    for run_num in range(RunNum):
        run_folder = "MapRun" + str(run_num+1) + ".feat/"
        print("Reading fMRI file from: ", run_folder)
        input_file_path = os.path.join(participant_fmri_folder, run_folder, inputf)

        if os.path.exists(input_file_path):
            fmri_data = nib.load(input_file_path).get_fdata(dtype=np.float32)
            # print some relevant metadata
            participant_fmri_data.append(fmri_data)
            print(f'There are {len(participant_fmri_data)} runs in total\n')
            print(f'N = {participant_fmri_data[0].shape[3]} TRs per run\n')
            print(f'The dimensions of the data for each run are: {participant_fmri_data[0].shape}\n')
            print(f'The stimulus duration is {stimdur} seconds\n')
            print(f'XYZ dimensionality is: {participant_fmri_data[0].shape[:3]} (one slice only in this example)\n')
            print(f'Numeric precision of data is: {type(participant_fmri_data[0][0,0,0,0])}\n')
            print(f"Participant {ParticipantID}, Run {run_num+1}: fMRI data shape = {fmri_data.shape}")
    return participant_fmri_data



# %%



# def CollectConds(ParticipantID):
#     participant_onset_folder = os.path.join(onsets_folder, ParticipantID)
#     participant_csr_designs = []
#     participant_stimuli_conds = set()
#     for run_onset in range(RunNum):
#         onset_file = f"{ParticipantID}_Run{run_onset+1}_onsets.txt"
#         onset_fname = os.path.join(participant_onset_folder, onset_file)
        
#         with open(onset_fname, 'r') as fh:
#             for line in fh:
#                 stim_id = line.split(" ")[0][1:]
#                 participant_stimuli_conds.add(stim_id)

#     participant_stimuli_conds = sorted(participant_stimuli_conds)

#     for run_onset in range(RunNum):
#         onset_file = f"{ParticipantID}_Run{run_onset+1}_onsets.txt"
#         onset_fname = os.path.join(participant_onset_folder, onset_file)
        
#         run_row, run_col, run_data = [], [], []
#         with open(onset_fname, 'r') as fh:
#             for line in fh:
#                 onset_time = round(float(line.split(" ")[1]) * (tr_old / tr_new)) 
#                 stim_id = line.split(" ")[0][1:]
#                 if stim_id in participant_stimuli_conds:
#                     col_index = participant_stimuli_conds.index(stim_id)
#                     run_row.append(onset_time)
#                     run_col.append(col_index)
#                     run_data.append(1)

#         num_conditions = len(participant_stimuli_conds)
#         run_csr = scipy.sparse.csr_matrix((run_data, (run_row, run_col)), shape=(1046, num_conditions))
#         participant_csr_designs.append(run_csr)
#         print(f"Participant {ParticipantID}, Run {run_onset+1}: Design matrix shape = {run_csr.shape}")


#     return participant_csr_designs


# def CollectFMRI(ParticipantID, inputf):
#     participant_fmri_data = []
    
#     participant_fmri_folder = os.path.join(mri_folder, ParticipantID)
#     for run_num in range(RunNum):
#         run_folder = f"MapRun{run_num+1}.feat/"
#         input_file_path = os.path.join(participant_fmri_folder, run_folder, inputf)
        
#         if os.path.exists(input_file_path):
#             fmri_data = nib.load(input_file_path).get_fdata(dtype=np.float32)
#             participant_fmri_data.append(fmri_data)
#             print(f"Participant {ParticipantID}, Run {run_num+1}: fMRI data shape = {fmri_data.shape}")
    
#     return participant_fmri_data


def RunGLMsingle(participant_csr_designs, participant_fmri_data, inputf, glm_single_folder_participant):
    opt = {
        'wantlibrary': 1,
        'wantglmdenoise': 1,
        'wantfracridge': 1,
        'wantfileoutputs': [1, 1, 1, 1],
        'wantmemoryoutputs': [0, 0, 0, 1],
        'sessionindicator': [1, 1, 1, 1, 2, 2, 2, 2]
    }
    
    glmsingle_obj = GLM_single(opt)
    start_time = time.time()
    participant_results_glmsingle = glmsingle_obj.fit(
        participant_csr_designs,
        participant_fmri_data,
        stimdur,
        tr_new,
        outputdir=glm_single_folder_participant
    )

    print(f"GLMsingle completed in {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")

## Main execution
if __name__ == "__main__":
    #print("Starting fMRI upsampling...")
    #Upsample(participant_list)
    
    print("\nStarting GLMsingle processing...")
    print(expid_mrid_map.items())
    for expid, participant_id in expid_mrid_map.items():
        ParticipantID = f"sub-{participant_id}"
        inputf = "upsampled_filtered_func_data_EF1reg.nii.gz"
        GLMsingleWrapper(expid, ParticipantID, inputf)
    
    print("Processing complete!")
