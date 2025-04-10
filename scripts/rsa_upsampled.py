# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 14:47:00 2025

@author: borib
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import scipy.sparse
import nibabel as nib
import rsatoolbox
import rsatoolbox.rdm as rsa_rdm
from rsatoolbox.data.dataset import Dataset
from rsatoolbox.rdm import calc_rdm, rdms, compare_kendall_tau
from rsatoolbox import vis, rdm
from rsatoolbox.model import ModelFixed
from rsatoolbox.inference import eval_bootstrap_pattern, bootstrap_sample_pattern
import pickle


mri_folder = "/mnt/alex/bids_format/derivatives/feat/ses-02/first_level/phase_mag"
onsets_folder = "/mnt/alex/bids_format/derivatives/onsets/ses-02_upsampled/"
outputdir_glmsingle = "/mnt/alex/bids_format/derivatives/upsample/ses-02/glmsingle/"

code_folder = "/mnt/alex/bids_format/code/"
masks_folder = "/mnt/alex/bids_format/derivatives/masks/masks_bori/"
atl_mask_folder = "/mnt/alex/bids_format/derivatives/masks/atl_masks"

output_folder = "/mnt/alex/bids_format/derivatives/rsa_ses02/correlations/eucl/"
if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
trials_folder = "/mnt/alex/bids_format/derivatives/rsa_ses02/rdms/"
if not os.path.exists(trials_folder):
    os.makedirs(trials_folder, exist_ok=True)
    
figure_folder = "/mnt/alex/bids_format/derivatives/rsa_ses02/figures/"
if not os.path.exists(figure_folder):
    os.makedirs(figure_folder, exist_ok=True)

psycholing_path = "/mnt/alex/bids_format/derivatives/rsa_ses02/stimuli/stimuli_v2_2_allcovariates.xlsx"
psychometric_path = "/mnt/alex/bids_format/derivatives/rsa_ses02/stimuli/stimuli_v2_psychometric_covariates.xlsx"
median_path = "/mnt/alex/bids_format/derivatives/rsa_ses02/stimuli/stimuli_v2_3_allcovariates_median.xlsx"
wordtovec_path = "/mnt/alex/bids_format/derivatives/rsa_ses02/stimuli/stimuli_v2_2_rsa_matrix_ids.xlsx"

# %%

def upper_tri(RDM):
    """Extract the upper triangular part of a square matrix (excluding the diagonal).

    Args:
        RDM (2Darray): A square matrix.

    Returns:
        1Darray: The upper triangular vector of the RDM.
    """
    m = RDM.shape[0]
    r, c = np.triu_indices(m, 1)
    return RDM[r, c]


# %%


#participant_list = ["sub-369463"]
#participant_list = open(os.path.join(code_folder, "subjects_ses-02.txt")).read().splitlines()

## ID corresp
expid_mrid_map = {}
with open(os.path.join(code_folder, "code_pairs_ses-02.txt"), "r") as file:
    next(file)
    for line in file:
        expid, participant_id = line.strip().split("\t")
        expid_mrid_map[expid] = participant_id

## Parameters
stimdur = 2
tr_old = 1.15
tr_new = 1
n_slices = 80
RunNum = 8
sumTR = 654

# %%

def CompareRDM(expid, participant_id, roi_paths):
    """
    Process a participant's data, compute RDMs for each ROI, and save outputs.
    """
    print(f"Processing participant: {participant_id}")
    
    print("Step 3: Computing betas RDMs for all ROIs...")
    rdm_results = ComputeBetasRDM(participant_id, roi_paths)

    print("Step 4: Saving visualizations...")
    SaveRDMVisualizations(rdm_results, participant_id)

    print(f"Finished processing participant: {participant_id}")
    EvaluateModelVsDataRDM(participant_id, roi_paths, rdm_results)

# %%

def CreatePsycholingRDM(expid, participant_id):
    """
    Process psycholing data for a single participant, merge with psychometric data, and generate RDMs.

    Args:
        participant_id (str): Participant ID (e.g., "sub-123").
        psycholing_path (str): Path to psycholing Excel file -1.
        psychometric_path (str): Path to psychometric data Excel file -1.
        median_path (str): Path to psychometric median data Excel file -1.
        wordtovec_path (str): Path to word-to-vector data Excel file -1.
        onsets_folder (str): Path to folder containing onset files -1 for each participant.
        output_folder ikyk
        figure_folder ikyk
        RunNum (int): Number of runs to process.

    Returns:
        dict: Filtered, concrete, and abstract dataframes for the participant.
    """
    # load psycholing data and create trial numbers
    psycholing_all = pd.read_excel(psycholing_path)
    psycholing_all['trialno'] = np.where(
        psycholing_all['stim_type'] == 'TARG', 'B' + psycholing_all['itemno'].astype(str),
        np.where(psycholing_all['stim_type'] == 'CLOSE', 'C' + psycholing_all['itemno'].astype(str),
        np.where(psycholing_all['stim_type'] == 'DISTANT', 'D' + psycholing_all['itemno'].astype(str), None))
    )

    psychometr_all = pd.read_excel(psychometric_path)
    psychometr_median = pd.read_excel(median_path)
    mördzs = pd.merge(psychometr_all, psychometr_median[['noun', 'median_split_mean']], on='noun')
    merge_psy = pd.merge(psycholing_all, mördzs, on=['adjective', 'noun'], how='outer', indicator=True)
    unmatched_df = merge_psy[merge_psy['_merge'] != 'both'].drop(columns=['_merge']).reset_index(drop=True)
    print("Unmatched Rows:\n", unmatched_df)

    wordtovec = pd.read_excel(wordtovec_path)

    participant_onset_folder = os.path.join(onsets_folder, expid)
    if not os.path.isdir(participant_onset_folder):
        raise ValueError(f"Folder for participant {participant_id, expid} not found in {onsets_folder}.")

    trialnos_in_txt = set()

    # process onset files in sequential order
    for run_onset in range(RunNum):
        onset_file = f"{expid}_Run{run_onset + 1}_onsets.txt"
        onset_fname = os.path.join(participant_onset_folder, onset_file)
        print(f"Reading onset file for stimlist: {onset_fname}")

        if os.path.isfile(onset_fname):
            with open(onset_fname, 'r') as fh:
                for line in fh:
                    trialnos_in_txt.add(line.split()[0])
        else:
            print(f"File {onset_fname} not found. Skipping this run.")

    psy_filtered = merge_psy[merge_psy['trialno'].isin(trialnos_in_txt)].copy()
    psy_filtered = psy_filtered.sort_values(by=['itemno', 'trialno'])
    trialnos_0 = psy_filtered.loc[psy_filtered['median_split_mean'] == 0, 'trialno'].tolist()
    trialnos_1 = psy_filtered.loc[psy_filtered['median_split_mean'] == 1, 'trialno'].tolist()

    # generate concrete and abstract dataframes 1-cosine
    conc = 1 - (wordtovec
            .loc[wordtovec['ID'].isin(trialnos_1)]
            .filter(items=['ID'] + trialnos_1, axis=1)
            .set_index('ID')
            .copy())
    abst = 1 - (wordtovec
            .loc[wordtovec['ID'].isin(trialnos_0)]
            .filter(items=['ID'] + trialnos_0, axis=1)
            .set_index('ID')
            .copy())

    psy_filtered.to_excel(f'{output_folder}/trials_{participant_id}.xlsx', index=False)
    conc.to_excel(f'{output_folder}/concrete_{participant_id}.xlsx', index=False)
    abst.to_excel(f'{output_folder}/abstract_{participant_id}.xlsx', index=False)

    # generate RDM visualizations
    plt.figure(figsize=(10, 8))
    plt.imshow(conc, cmap='viridis', vmin=0, vmax=1)
    plt.colorbar(label='Dissimilarity')
    plt.xticks(range(len(conc)), conc.index, rotation=90)
    plt.yticks(range(len(conc)), conc.index)
    plt.title(f"RDM (Median Split) for Concrete - Participant {participant_id}")
    plt.savefig(f'{figure_folder}/rdm_median_split_conc_v2_{participant_id}.jpg')
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.imshow(abst, cmap='hot', vmin=0, vmax=1)
    plt.colorbar(label='Dissimilarity')
    plt.xticks(range(len(abst)), abst.index, rotation=90)
    plt.yticks(range(len(abst)), abst.index)
    plt.title(f"RDM (Median Split) for Abstract - Participant {participant_id}")
    plt.savefig(f'{figure_folder}/rdm_median_split_abstr_v2_{participant_id}.jpg')
    plt.close()

    return {"trialnos_0": trialnos_0, "trialnos_1": trialnos_1, "filtered": psy_filtered, "concrete": conc, "abstract": abst}
# %%



def SaveRDMVisualizations(rdm_results, participant_id):
    """
    Save RDM visualizations for a participant with consistent formatting.

    Args:
        rdm_results (dict): Dictionary containing RDMs for each ROI.
        participant_id (str): The participant's ID.
    """
    fig_size = (8, 6)
    vmin, vmax = 0, 1  # Normalize the color scale for all RDMs

    for roi_name, roi_rdms in rdm_results.items():
        for rdm_type, rdm_data in roi_rdms.items():
            fig, ax = plt.subplots(figsize=fig_size)

            if isinstance(rdm_data, rsatoolbox.rdm.RDMs):
                rdm_matrix = rdm_data.get_matrices()[0]
            else:
                rdm_matrix = np.array(rdm_data)

            im = ax.imshow(rdm_matrix, cmap='viridis', vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label("Dissimilarity", fontsize=12)

            ax.set_title(f"{roi_name.upper()} {rdm_type.upper()} RDM - {participant_id}", fontsize=14)
            ax.set_xlabel("Conditions", fontsize=12)
            ax.set_ylabel("Conditions", fontsize=12)

            ax.set_xticks(range(rdm_matrix.shape[0]))
            ax.set_yticks(range(rdm_matrix.shape[1]))
            ax.set_xticklabels(range(1, rdm_matrix.shape[0] + 1), fontsize=8, rotation=90)
            ax.set_yticklabels(range(1, rdm_matrix.shape[1] + 1), fontsize=8)

            save_path = os.path.join(figure_folder, f"{roi_name}_{rdm_type}_neural_rdm_{participant_id}.jpg")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)

            print(f"Saved RDM visualization: {save_path}")


# %%

def CollectConds(expid, participant_id):
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
    
    participant_designs = []
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
        participant_designs.append([run_row, run_col, run_data])
        
    participant_allruns_design = [y for x in participant_designs for y in x[1]]
    participant_allruns_design_guide=[[x,y] for x,y in zip(range(len(participant_allruns_design)),participant_allruns_design)]
    participant_allruns_design_guide = sorted(participant_allruns_design_guide, key=lambda x: x[1])
    participant_allruns_trialorder = [x[0] for x in participant_allruns_design_guide]
    
    print(f"Participant {participant_id}, Experimental ID {expid}, Run {run_onset+1}: Design matrix shape = {run_csr.shape}")
    return {"participant_csr_designs": participant_csr_designs, "participant_stimuli_conds": participant_stimuli_conds, "participant_allruns_trialorder": participant_allruns_trialorder}

# %%

def ComputeBetasRDM(participant_id, roi_paths):
    """
    Compute RDMs for GLMsingle results and separate RDMs per ROI.

    Args:
        participant_id (str): Participant ID.
        roi_paths (dict): Dictionary of ROI paths, e.g.,
                          {"temporal_pole": "/path/to/temporal_pole_mask.nii.gz"}

    Returns:
        dict: RDMs for each ROI.
    """
    trial_data = CreatePsycholingRDM(expid, participant_id)
    cond_data = CollectConds(expid, participant_id)

    trialnos_0 = trial_data["trialnos_0"]
    trialnos_1 = trial_data["trialnos_1"]
    participant_stimuli_conds = cond_data["participant_stimuli_conds"]
    participant_allruns_trialorder = cond_data["participant_allruns_trialorder"]

    # GLMsingle betas
    results_glmsingle = np.load(os.path.join(outputdir_glmsingle, participant_id, 'upsampled_filtered_func_data_EF1reg/TYPED_FITHRF_GLMDENOISE_RR.npy'), allow_pickle=True).item()
    betas = results_glmsingle['betasmd']
    betas_reorder = betas[:, :, :, participant_allruns_trialorder]

    # average betas across triplets
    betas_average = np.zeros([betas_reorder.shape[0], betas_reorder.shape[1], betas_reorder.shape[2], len(participant_stimuli_conds)])
    for stim_trio in range(len(participant_stimuli_conds)):
        betas_average[:, :, :, stim_trio] = np.nanmean(betas_reorder[:, :, :, stim_trio*3:(stim_trio+1)*3], axis=3)
    betas_average = np.nan_to_num(betas_average)

    rdm_results = {}

    # compute RDMs per ROI
    for roi_name, roi_path in roi_paths.items():
        if os.path.exists(roi_path):
            print(f"Applying mask for ROI: {roi_name}")
            roi_img = nib.load(roi_path)
            roi_data = roi_img.get_fdata().astype(float)
            roi_data[roi_data == 0] = np.nan
            masked_betas = betas_average * roi_data[:, :, :, np.newaxis]

            betas_transposed = np.transpose(masked_betas, (3, 0, 1, 2))
            data_2d = betas_transposed.reshape([betas_transposed.shape[0], -1])
            data_2d = np.nan_to_num(data_2d)

            trial_indices_conc = np.where(np.isin(participant_stimuli_conds, trialnos_1))[0]
            trial_indices_abst = np.where(np.isin(participant_stimuli_conds, trialnos_0))[0]

            data_conc = data_2d[trial_indices_conc, :]
            data_abst = data_2d[trial_indices_abst, :]

            concr_dataset = rsatoolbox.data.base.DatasetBase(
                measurements=data_conc,
                descriptors={'condition': 'Concrete'},
                obs_descriptors={'trialno': trialnos_1}
            )
            abstr_dataset = rsatoolbox.data.base.DatasetBase(
                measurements=data_abst,
                descriptors={'condition': 'Abstract'},
                obs_descriptors={'trialno': trialnos_0}
            )

            rdm_conc = calc_rdm(concr_dataset)
            rdm_abst = calc_rdm(abstr_dataset)

            rdm_conc.to_df().to_csv(os.path.join(trials_folder, f"{participant_id}_{roi_name}_data_rdm_concrete.csv"), index=False)
            rdm_abst.to_df().to_csv(os.path.join(trials_folder, f"{participant_id}_{roi_name}_data_rdm_abstract.csv"), index=False)
            print(f"Saved data RDMs to CSV for ROI {roi_name} of participant {participant_id}")

            rdm_results[roi_name] = {"rdm_conc": rdm_conc, "rdm_abst": rdm_abst}

    return rdm_results

# %%
def EvaluateModelVsDataRDM(participant_id, roi_paths, rdm_results):
    """
    Creates model RDMs from psycholing data and evaluates them against data RDMs derived from GLMsingle betas
    separately for each ROI. Saves results as .pkl files.

    Args:
        participant_id (str): Participant ID.
        roi_paths (dict): Dictionary of ROI paths, e.g.,
                          {"temporal_pole": "/path/to/temporal_pole_mask.nii.gz"}

    Returns:
        dict: Evaluation and correlation results for each ROI.
    """
    print(f"Processing participant: {participant_id}")

    print("Step 1: Creating model RDMs...")
    trial_data = CreatePsycholingRDM(expid, participant_id)
    trialnos_1 = trial_data["trialnos_1"]
    trialnos_0 = trial_data["trialnos_0"]
    print(trialnos_1)
    print(trial_data["concrete"].index)
    
    concrete_matrix = np.array(trial_data["concrete"])
    abstract_matrix = np.array(trial_data["abstract"])
    
    print(f"Concrete matrix shape: {concrete_matrix.shape}")
    print(f"Abstract matrix shape: {abstract_matrix.shape}")
    
    concrete_vector = upper_tri(concrete_matrix)
    abstract_vector = upper_tri(abstract_matrix)
    
    print(f"Concrete vector shape: {concrete_vector.shape}")
    print(f"Abstract vector shape: {abstract_vector.shape}")
    

    # prepare model RDMs from psycholing data
    model_rdm_conc = rsatoolbox.rdm.RDMs(
        dissimilarities=concrete_vector,
        pattern_descriptors={'trialno': trialnos_1}
    )
    model_rdm_abst = rsatoolbox.rdm.RDMs(
        dissimilarities=abstract_vector,
        pattern_descriptors={'trialno': trialnos_0}
    )
    
    model_rdm_conc.to_df().to_csv(os.path.join(output_folder, f"{participant_id}_model_rdm_concrete.csv"), index=False)
    model_rdm_abst.to_df().to_csv(os.path.join(output_folder, f"{participant_id}_model_rdm_abstract.csv"), index=False)
    print(f"Saved model RDMs to CSV for participant {participant_id}")

    

    all_results = {}

    for roi_name, roi_rdm_data in rdm_results.items():
        print(f"Step 3: Evaluating RDMs for ROI: {roi_name}")

        data_rdm_conc = roi_rdm_data["rdm_conc"]
        data_rdm_abst = roi_rdm_data["rdm_abst"]
        
        model_fixed_conc = rsatoolbox.model.ModelFixed('Concrete_Model', model_rdm_conc)
        model_fixed_abst = rsatoolbox.model.ModelFixed('Abstract_Model', model_rdm_abst)
        
        eval_conc = rsatoolbox.inference.evaluate.eval_bootstrap_pattern(models=model_fixed_conc, 
        data=data_rdm_conc, method='corr', N=1000, pattern_descriptor='trialno')
        
        correlations_conc = rsatoolbox.rdm.compare_kendall_tau(model_rdm_conc, data_rdm_conc)

        # save results
        concrete_results = {
            "correlations": correlations_conc,
            "evaluation" : eval_conc
        }
        concrete_file = os.path.join(output_folder, f"{participant_id}_{roi_name}_concrete_results.pkl")
        with open(concrete_file, 'wb') as f:
            pickle.dump(concrete_results, f)
        print(f"Concrete results saved to {concrete_file}")
        
        
        eval_abst = rsatoolbox.inference.evaluate.eval_bootstrap_pattern(models=model_fixed_abst, 
        data=data_rdm_abst, method='corr', N=1000, pattern_descriptor='trialno')
        
        correlations_abst = rsatoolbox.rdm.compare_kendall_tau(model_rdm_abst, data_rdm_abst)

        abstract_results = {
            "correlations": correlations_abst,
            "evaluation" : eval_abst
        }
        
        abstract_file = os.path.join(output_folder, f"{participant_id}_{roi_name}_abstract_results.pkl")
        with open(abstract_file, 'wb') as f:
            pickle.dump(abstract_results, f)
        print(f"Abstract results saved to {abstract_file}")

        # store ROI results in dictionary
        all_results[roi_name] = {
            "concrete": concrete_results,
            "abstract": abstract_results
        }

    print("Evaluation complete for all ROIs.")

    return all_results

# %%

for expid, participant_id in expid_mrid_map.items():
    participant_id = f"sub-{participant_id}"
    
    roi_paths = {
        "temporal_pole": os.path.join(masks_folder, f"{participant_id}", "temporal_pole_binary_func.nii.gz"),
        "angular_gyrus": os.path.join(masks_folder, f"{participant_id}", "ag_binary_func.nii.gz"),
        "fusiform_gyrus": os.path.join(masks_folder, f"{participant_id}", "fusiform_binary_func.nii.gz"),
        "inferior_frontal_gyrus": os.path.join(masks_folder, f"{participant_id}", "ifg_binary_func.nii.gz"),
        "inferior_temporal_gyrus": os.path.join(masks_folder, f"{participant_id}", "itg_binary_func.nii.gz"),
        "lateral_occ_cortex": os.path.join(masks_folder, f"{participant_id}", "loc_binary_func.nii.gz"),
        "parahippocampal_gyrus": os.path.join(masks_folder, f"{participant_id}", "phg_binary_func.nii.gz"),
        "superior_temporal_gyrus": os.path.join(masks_folder, f"{participant_id}", "stg_binary_func.nii.gz"),
        "juxtapos_lobule_cortex": os.path.join(masks_folder, f"{participant_id}", "juxtapositional-lob-binary_func.nii.gz"),
        "postcentral_gyrus": os.path.join(masks_folder, f"{participant_id}", "postcentral_gyrus_binary_func.nii.gz"),
        "precentral_gyrus": os.path.join(masks_folder, f"{participant_id}", "precentral_gyrus_binary_func.nii.gz"),
        "anterior_temporal_lobe": os.path.join(atl_mask_folder, f"{participant_id}_atl_mask_in_func_space.nii.gz"),
        "middle temporal_gyrus": os.path.join(masks_folder, f"{participant_id}", "mtg_binary_func.nii.gz")
    }

    try:
        CompareRDM(expid, participant_id, roi_paths)
        print(f"Finished with participant {participant_id}")
        
    except Exception as e:
        print(f"Error occurred with participant {participant_id}: {e}. Skipping to the next participant.")

    