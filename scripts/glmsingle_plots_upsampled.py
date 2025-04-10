# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 22:24:37 2025

@author: borib
"""

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


mri_folder = "/mnt/alex/bids_format/derivatives/feat/ses-02/first_level/phase_mag"
onsets_folder = "/mnt/alex/bids_format/derivatives/onsets/ses-02_upsampled/"
glm_single_folder = "/mnt/alex/bids_format/derivatives/upsample/ses-02/glmsingle/"
if not os.path.exists(glm_single_folder):
    os.makedirs(glm_single_folder, exist_ok=True)
out_folder = "/mnt/alex/bids_format/derivatives/upsample/ses-02/"
code_folder = "/mnt/alex/bids_format/code/"
glm_plot_folder = "/mnt/alex/bids_format/derivatives/upsample/ses-02/glmsingle/plots/"
if not os.path.exists(glm_plot_folder):
    os.makedirs(glm_plot_folder, exist_ok=True)


expid_mrid_map = {}
with open(os.path.join(code_folder, "code_pairs_ses-02.txt"), "r") as file:
    next(file)
    for line in file:
        expid, participant_id = line.strip().split("\t")
        expid_mrid_map[expid] = participant_id

# %%

# parameters
stimdur = 2
tr_old = 1.15
tr_new = 1
n_slices = 80
RunNum = 8
sumTR = 654
# %%

def GLMplotWrapper(ParticipantID, expid, inputf):
    glm_single_folder_participant = os.path.join(glm_single_folder, ParticipantID, inputf.split('.')[0])
    participant_csr_designs = CollectConds(ParticipantID, expid)
    participant_fmri_data = CollectFMRI(ParticipantID, inputf)
    results_glmsingle = CollectGLMSingle(ParticipantID, inputf)
    
    DiagnosticFigures(ParticipantID, participant_fmri_data, participant_csr_designs, results_glmsingle, inputf)


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

# load existing file outputs if they exist
def CollectGLMSingle(ParticipantID, inputf):
  outputdir_glmsingle = os.path.join(glm_single_folder, ParticipantID, inputf.split('.')[0])

  results_glmsingle = dict()
  results_glmsingle['typea'] = np.load(os.path.join(outputdir_glmsingle,'TYPEA_ONOFF.npy'),allow_pickle=True).item()
  results_glmsingle['typeb'] = np.load(os.path.join(outputdir_glmsingle,'TYPEB_FITHRF.npy'),allow_pickle=True).item()
  results_glmsingle['typec'] = np.load(os.path.join(outputdir_glmsingle,'TYPEC_FITHRF_GLMDENOISE.npy'),allow_pickle=True).item()
  results_glmsingle['typed'] = np.load(os.path.join(outputdir_glmsingle,'TYPED_FITHRF_GLMDENOISE_RR.npy'),allow_pickle=True).item()

  return results_glmsingle

def DiagnosticFigures(ParticipantID, participant_fmri_data, participant_csr_designs, results_glmsingle, inputf):
  inputtype = inputf.split("_")[-1]

  # Figure 1. - 4 volumes
  plot_fields = ['betasmd','R2','HRFindex','FRACvalue']
  colormaps = ['RdBu_r','hot','jet','copper']
  clims = [[-5,5],[0,85],[0,20],[0,1]]
  xyzt = participant_fmri_data[0].shape
  xyz = xyzt[:3]

  meanvol = np.squeeze(np.mean(participant_fmri_data[0].reshape(xyzt),3))
  brainmask = meanvol > 275

  plt.figure(figsize=(12,8))

  for i in range(len(plot_fields)):

      plt.subplot(2,2,i+1)

      if i == 0:
          # when plotting betas, for simplicity just average across all image presentations
          # this will yield a summary of whether voxels tend to increase or decrease their
          # activity in response to the experimental stimuli (similar to outputs from
          # an ONOFF GLM)
          plot_data = np.nanmean(results_glmsingle['typed'][plot_fields[i]].reshape(104,104,56,504),3)[55,:,:].T
          titlestr = 'average GLM betas (localizer runs 1-4)'

      else:
          # plot all other voxel-wise metrics as outputted from GLMsingle
          plot_data = results_glmsingle['typed'][plot_fields[i]].reshape(104,104,56)[55,:,:].T
          titlestr = plot_fields[i]

      #plot_data[~brainmask] = np.nan # remove values outside the brain for visualization purposes
      plt.imshow(plot_data, cmap=colormaps[i],clim=clims[i], origin="lower")
      plt.colorbar()
      plt.title(titlestr)
      plt.axis(False)

  plt.tight_layout()
  plt.savefig(glm_plot_folder + ParticipantID + "_" + inputtype + "_glmsingle_betas.png", dpi=300)

  # reliability plot
  nblocks = 8
  xyzn = (xyz[0],xyz[1],xyz[2],nblocks)
  # consolidate design matrices
  designALL = np.concatenate(participant_csr_designs, axis=0)
  models = dict()
  #models['assumehrf'] = results_glmsingle['typeb']['betasmd'].reshape(xyzn)
  models['fithrf'] = results_glmsingle['typeb']['betasmd']
  models['fithrf_glmdenoise'] = results_glmsingle['typec']['betasmd']
  models['fithrf_glmdenoise_rr'] = results_glmsingle['typed']['betasmd']

  # construct a vector containing 0-indexed condition numbers in chronological order
  corder = []
  for p in range(designALL.shape[0]):
      if np.any(designALL[p]):
          corder.append(np.argwhere(designALL[p])[0,0])

  corder = np.array(corder)

  repindices = []

  for p in range(designALL.shape[1]): # loop over every condition
      temp = np.argwhere(corder==p)[:,0] # find indices where this condition was shown
      if len(temp) >= 2:
          repindices.append(temp)

  repindices = np.vstack(np.array(repindices)).T

  print(f'There are {repindices.shape[1]} repeated conditions in the experiment\n')
  print(f'There are {repindices.shape[0]} instances of each repeated condition across 4 runs\n')
  print(f'Betas from blocks containing the first localizer condition can be found at the following indices of GLMsingle output beta matrices:\n\n{repindices[:,0]}')

  vox_reliabilities = [] # output variable for reliability values

  modelnames = list(models.keys())
  n_cond = repindices.shape[1]

  # for each beta version
  for m in range(len(modelnames)):
    print(f'computing reliability for beta version: {modelnames[m]}')
    time.sleep(1)

    # organize the betas by (X,Y,Z,repeats,conditions) using the repindices variable
    betas = models[modelnames[m]][:,:,:,repindices]
    x,y,z = betas.shape[:3]

    # create output volume for voxel reliability scores
    rels = np.full((x,y,z),np.nan)

    # loop through voxels in the 3D
    for xx in tqdm(range(x)):
        for yy in range(y):
            for zz in range(z):
                # for this voxel, get beta matrix of (repeats,conditions)
                vox_data = betas[xx,yy,zz]

                # average odd and even betas after shuffling
                even_data = np.nanmean(vox_data[::2],axis=0)
                odd_data = np.nanmean(vox_data[1::2],axis=0)

                # reliability at a given voxel is pearson correlation between the
                # odd- and even-presentation beta vectors
                rels[xx,yy,zz] = np.corrcoef(even_data,odd_data)[1,0]

    vox_reliabilities.append(rels)

  comparison = []
  for vr in vox_reliabilities:
      comparison.append(np.nanmedian(vr))
  #comparison = np.vstack(comparison)

  plt.figure(figsize=(18,6))
  plt.subplot(121)
  plt.bar(np.arange(len(comparison)),comparison,width=0.4)
  plt.ylim([0,0.7])
  plt.title('Median voxel split-half reliability of GLM models')
  plt.xticks(np.arange(3),np.array(['FITHRF', 'FITHRF\nGLMDENOISE', 'FITHRF\nGLMDENOISE\nRR']));

  plt.tight_layout()
  plt.savefig(glm_plot_folder + ParticipantID + "_" + inputtype + "_glmsingle_mediansplit.png", dpi=300)

  # we can also look at how distributions of FFA/V1 voxel reliabilities change
  # between the baseline GLM and the final output of GLMsingle (fithrf+GLMdenoise+RR)
  plt.subplot(122)
  plt.hist(vox_reliabilities[0].reshape(-1),25,alpha=0.6,color='tomato');
  plt.hist(vox_reliabilities[1].reshape(-1),25,alpha=0.6,color='limegreen');
  plt.hist(vox_reliabilities[2].reshape(-1),25,alpha=0.6,color='gold');
  plt.xlabel('reliability (r)')
  plt.ylabel('# voxels')
  plt.legend(['FITHRF', 'FITHRF\nGLMDENOISE', 'FITHRF\nGLMDENOISE\nRR'])
  plt.title('Change in distribution of voxel reliabilities\ndue to GLMsingle');

  plt.tight_layout()
  plt.savefig(glm_plot_folder + ParticipantID + "_" + inputtype + "_glmsingle_reliability_distr.png", dpi=300)


for expid, participant_id in expid_mrid_map.items():
    ParticipantID = f"sub-{participant_id}"
    inputf = "upsampled_filtered_func_data_EF1reg.nii.gz"
    print(f"Processing Participant: {ParticipantID}")
    try:
        GLMplotWrapper(ParticipantID, expid, inputf)
    except Exception as e:
        print(f"    Error processing {ParticipantID} with {inputf}: {e}")