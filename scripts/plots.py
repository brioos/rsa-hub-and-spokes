# -*- coding: utf-8 -*-
"""
Created on Sun Mar 16 20:14:25 2025

@author: borib
"""

# %% ROI VISUALISATION

# to be done: add legend to plot
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from nilearn import image, plotting
from matplotlib.colors import ListedColormap

mask_dir = "../masks/binary/"
mni_template = datasets.load_mni152_template()
roi_paths = glob.glob(os.path.join(mask_dir, "*_binary.nii.gz"))

# extract ROI names from filenames
roi_names = [os.path.basename(p).replace("_binary.nii.gz", "") for p in roi_paths]

# get dims
example_img = image.load_img(roi_paths[0])
combined_data = np.zeros(example_img.shape)

# unique value for each roi - should be unified for later aggregated regions
for i, mask_path in enumerate(roi_paths):
    mask_img = image.load_img(mask_path)
    combined_data[mask_img.get_fdata() > 0] = i + 1
colored_rois_img = image.new_img_like(example_img, combined_data)

color_list = plt.cm.jet(np.linspace(0, 1, len(roi_names)))  
custom_cmap = ListedColormap(color_list)

plotting.plot_roi(colored_rois_img, bg_img=mni_template, title="Anatómialilag definiált ROI-k", cmap=custom_cmap, cut_coords = (50, -20, -15))

fig, ax = plt.subplots(figsize=(6, len(roi_names) // 2))
for i, roi_name in enumerate(roi_names):
    ax.scatter([], [], c=[color_list[i]], label=roi_name)

ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
ax.axis("off")
plt.title("ROI Color Mapping")
plt.show()
