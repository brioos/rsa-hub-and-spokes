# -*- coding: utf-8 -*-
"""
@author: borib
"""
# %% LIBRARY

import os
import pickle
import pandas as pd
import numpy as np

directory = "../correlations/eucl/"
figures = "../figures/result_figures/"
if not os.path.exists(figures):
    os.makedirs(figures, exist_ok=True)

# %% LOAD DATA


# listing all result files
pkl_files = [f for f in os.listdir(directory) if f.endswith("_results.pkl")]


results_data = []

# specifying hypothetized ROI-s
ling = ["inferior frontal", "angular gyrus", "superior temporal"]
vis = ["inferior temporal", "lateral occ", "precentral gyrus", "postcentral gyrus"]


for pkl_file in pkl_files:
    file_path = os.path.join(directory, pkl_file)
    
    # extracting data from filename
    parts = pkl_file.replace("_results.pkl", "").split("_")
    
    subject_id = parts[0]
    roi_name = roi_name = f"{parts[1]} {parts[2]}"
    condition = parts[-1]
    if roi_name in ling:
        roi_type = "nyelvi-auditoros"
        
    elif roi_name in vis:
        roi_type = "szenzoromotoros-vizuális"
        
    else:
        roi_type = "transzmodális"

    # extracting data from file
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    
    correlation_value = data["evaluation"].get_means()
    p_zero = data["evaluation"].test_zero()
    
    # fisher z correlations
    fisher_z = np.arctanh(correlation_value[0])

    
    # appending data to list
    results_data.append({
        "Subject": subject_id,
        "ROI": roi_name,
        "Condition": condition,
        "Type": roi_type,
        "Correlation": correlation_value[0],
        "Fisher" : fisher_z,
        "Eval": p_zero[0]
    })
    print(f"Processed {pkl_file}")

# list to df
df_results = pd.DataFrame(results_data)

# checking results
df_results["Subject"].nunique()
df_results["Correlation"].mean() #0.01

df_results[df_results["Condition"] == "concrete"]["Correlation"].mean() #0.008
df_results[df_results["Condition"] == "abstract"]["Correlation"].mean() #0.003

df_results["ROI"].nunique()

# %% RM ANOVA

import pingouin as pg

# repeated measures anova on the fisher z transformed correlations
res = pg.rm_anova(dv='Fisher', within=['Type','Condition'], subject='Subject', data=df_results, detailed=True)
print(res) # condition effect sig p=0.042

#post-hocs
post_hocs = pg.pairwise_ttests(dv='Fisher', within=['Type','Condition'], subject='Subject', padjust='fdr_bh', data=df_results)
print(post_hocs)

# saving outputs
output_path = "../correlations/anova_type_results.xlsx"

with pd.ExcelWriter(output_path) as writer:
    res.to_excel(writer, sheet_name="ANOVA", index=False)
    post_hocs.to_excel(writer, sheet_name="Post-hoc tests", index=False)
    

# %% SPHERICITY

# assumption check of sphericity
pg.sphericity(data=df_results, dv='Correlation', within=['Type','Condition'], subject='Subject')

# %% NORMALITY

# assumption check of normality
pg.normality(data=df_results, dv='Correlation', group='Condition')

# %% GROUP LEVEL VISUALISATION

import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

# calculating mean correlation
mean_correlation = df_results.groupby(['Type', 'Condition'])['Correlation'].mean().reset_index().sort_values(by='Correlation', ascending=False)

with plt.style.context('high-contrast'):
    plt.figure(figsize=(12, 6))
    sns.barplot(data=mean_correlation, x='Type', y='Correlation', hue='Condition', errorbar=None)

    plt.xticks(rotation=45, ha='right')
    plt.xlabel("ROI")
    plt.ylabel("Átlagos korreláció")
    plt.legend(title="Feltétel")
    plt.tight_layout()
    plt.show()



# %% ONE SAMPLE T TEST

from scipy.stats import ttest_1samp

# tároljuk az eredményeket
t_test_results = []

# végigmegyünk az összes ROI-n és minden feltételen
for roi in df_results["ROI"].unique():
    for condition in df_results["Condition"].unique():
        # kiválasztjuk az adott ROI és feltétel adatait
        subset = df_results[(df_results["ROI"] == roi) & (df_results["Condition"] == condition)]
        
        # kiszűrjük a Fisher z értékeket
        fisher_values = subset["Fisher"].values
        
        # egymintás t-teszt nullhipotézissel, hogy az átlag = 0
        t_stat, p_value = ttest_1samp(fisher_values, popmean=0)
        
        t_test_results.append({
            "ROI": roi,
            "Condition": condition,
            "N": len(fisher_values),
            "T-value": t_stat,
            "p-value": p_value
        })

# DataFrame-be rakjuk az eredményeket
df_ttest_cond = pd.DataFrame(t_test_results)

# p-érték korrekció (pl. FDR - Benjamini-Hochberg)
from statsmodels.stats.multitest import multipletests
df_ttest_cond["p-corrected"] = multipletests(df_ttest_cond["p-value"], method="fdr_bh")[1]

# with pd.ExcelWriter(output_path) as writer:
#     df_ttest.to_excel(writer, sheet_name="ttest", index=False)
#     df_ttest_cond.to_excel(writer, sheet_name="ttest_cond", index=False)
    
    
# %% PAIRED T TEST ROI

from scipy.stats import ttest_rel

# where at least one of the conditions had significant correlations
significant_rois = df_ttest_cond[df_ttest_cond["p-corrected"] < 0.05]["ROI"].unique()
paired_results = []

for roi in significant_rois:
    # abstract-concrete for each participant
    pivot = df_results[df_results["ROI"] == roi].pivot(index="Subject", columns="Condition", values="Fisher")
    
    if "abstract" in pivot.columns and "concrete" in pivot.columns:
        t_stat, p_value = ttest_rel(pivot["abstract"], pivot["concrete"])
        
        paired_results.append({
            "ROI": roi,
            "T-value": t_stat,
            "p-value": p_value,
            "N": len(pivot)
        })

df_paired = pd.DataFrame(paired_results)

# mutiple comparisons
from statsmodels.stats.multitest import multipletests
df_paired["p-corrected"] = multipletests(df_paired["p-value"], method="fdr_bh")[1]


# %% PAIRED T TEST ROI TYPE



# filter df_results to only include significant ROIs
filtered_df = df_results[df_results["ROI"].isin(significant_rois)]

# participant-wise mean fisher z within each ROI type and condition
averaged = filtered_df.groupby(["Subject", "Type", "Condition"])["Fisher"].mean().reset_index()

# wide format
pivot = averaged.pivot(index=["Subject", "Type"], columns="Condition", values="Fisher").reset_index()

# paired t-tests per ROI type
typewise_results = []
for roi_type in pivot["Type"].unique():
    subset = pivot[pivot["Type"] == roi_type]
    
    if "abstract" in subset.columns and "concrete" in subset.columns:
        t_stat, p_value = ttest_rel(subset["abstract"], subset["concrete"])
        typewise_results.append({
            "ROI Type": roi_type,
            "T-value": t_stat,
            "p-value": p_value,
            "N": len(subset)
        })

# multiple comparisons
df_typewise_ttest = pd.DataFrame(typewise_results)
df_typewise_ttest["p-corrected"] = multipletests(df_typewise_ttest["p-value"], method="fdr_bh")[1]


# %%

from scipy.stats import sem
import scienceplots

plot_data = averaged.groupby(["Type", "Condition"])["Fisher"].agg(["mean", sem]).reset_index()
plot_data.rename(columns={"mean": "Mean", "sem": "SEM"}, inplace=True)
plot_data["Condition"] = plot_data["Condition"].map({
    "concrete": "Konkrét",
    "abstract": "Absztrakt"
})

# merge with significance info
sig_map = dict(zip(df_typewise_ttest["ROI Type"], df_typewise_ttest["p-corrected"]))
plot_data["p-corrected"] = plot_data["Type"].map(sig_map)


import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("science")
roi_types = plot_data["Type"].unique()
file_paths = []

for roi_type in roi_types:
    data = plot_data[plot_data["Type"] == roi_type]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.barplot(data=data, x="Type", y="Mean", hue="Condition", palette="muted", ci=None)

    for i, row in data.iterrows():
        x_pos = 0.2 if row["Condition"] == "Konkrét" else -0.2
        ax.errorbar(x=0 + x_pos, y=row["Mean"], yerr=row["SEM"], fmt='none', color='black', capsize=5)

    star = row["Significance"]
    if star:
        y_max = data["Mean"].max()
        ax.text(0.15, y_max, star, ha='center', va='bottom', color='black', fontsize=18)

    ax.set_title(roi_type)
    ax.set_xlabel("")
    ax.set_ylabel("Átlagos Fisher z-transzformált korreláció")
    ax.set_xticks([])
    ax.legend(title="Feltétel")

    plt.tight_layout()
    file_path = os.path.join(figures, f"{roi_type.replace(' ', '_')}_tbarplot.png")
    fig.savefig(file_path, dpi=300)
    file_paths.append(file_path)
plt.close('all')


# %% FIRST LEVEL VISUALISATION X

import matplotlib.pyplot as plt
import seaborn as sns

# creating condition x ROI
df_results["Cond_ROI"] = df_results["Condition"] + " | " + df_results["Type"]

# p values to significance markers
df_results["Significance"] = df_results["Eval"].apply(lambda p: "*" if p < 0.05 else "")
with plt.style.context('high-contrast'):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        x=df_results["Cond_ROI"],
        y=df_results["Correlation"],
        hue=df_results["Significance"],
        edgecolor="black",
        s=50
    )
    
    # labeling
    for i, row in df_results.iterrows():
        if row["Significance"]:
            plt.text(row["Cond_ROI"], row["Correlation"], row["Significance"], fontsize=12, ha='right')
    
    plt.axhline(0, color='black', linestyle='--', linewidth=1)  # reference line at 0 correlation
    plt.xlabel("Feltétel | ROI")
    plt.ylabel("Korreláció")
    plt.xticks(rotation=90) 
    plt.show()
    
# %% FIRST LEVEL SIG CORRELATIONS X

roi_counts = df_results.groupby("Cond_ROI").size()
roi_significant_counts = df_results[df_results["Eval"] < 0.05].groupby("Cond_ROI").size()

# percentage of sig correlations per roi
roi_significance_percentage = (roi_significant_counts / roi_counts * 100).fillna(0)

df_significance_percentage = roi_significance_percentage.reset_index()
df_significance_percentage.columns = ["Cond_ROI", "Significance_Percentage"]


# %% BINOMIAL REGRESSION? ALTERNATIVE X

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


df_counts = roi_significant_counts.reset_index()
df_counts.columns = ["Cond_ROI", "Significant"]
df_counts["Total"] = 25
df_counts["Failures"] = df_counts["Total"] - df_counts["Significant"]

df_counts[["Condition", "ROI"]] = df_counts["Cond_ROI"].str.split(r"\|", expand=True)

# binomial
model = smf.glm(
    formula='Significant + Failures ~ Condition * ROI',
    data=df_counts,
    family=sm.families.Binomial()
)

result = model.fit()
print(result.summary())

res_bim = result.summary2().tables[1].reset_index()
