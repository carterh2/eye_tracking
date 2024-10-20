#!/usr/bin/env python
# coding: utf-8

# # License
# nemo_eyetracking
# Copyright (C) 2022 Utrecht University
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter, zoom
from tqdm import tqdm

from saleval import *


def run_analysis(fixations, n_fix):
    pd.set_option("mode.chained_assignment", None)
    show = False  # set show=False to avoid displaying results

    # initialise Path objects to facilitate loading and saving of results
    image_path = Path.cwd() / "images"
    results_path = Path.cwd() / "results"
    smaps_path = image_path / "saliency_maps"

    repo_path = Path.cwd().parent
    n_fix_path = repo_path / 'num_fix' / str(n_fix)
    
    if not Path.exists(n_fix_path):
        Path.mkdir(n_fix_path)

    pinfo = pd.read_csv(
        repo_path / 'results' / "participant_info.csv",
        usecols=["ID", "Gender", "DoB", "Valid", "Valid Freeviewing"],
    )
    free_viewing_gray = load_smap(image_path / "free_viewing_img_1080.png")

    # remove leading and trailing whitespaces for consistency in Gender column and
    # add age column to participant information
    pinfo["Gender"] = pinfo.Gender.str.strip()
    pinfo["DoB"] = pinfo["DoB"].astype(int)

    pinfo["CurrYear"] = pinfo["ID"].apply(lambda x: str(x)[:4]).astype(int)
    pinfo["CurrMonth"] = pinfo["ID"].apply(lambda x: str(x)[4:6]).astype(int)

    # Because we don't know pp's birth month (only year), we
    # just assume that everyone's birthday is halfway through the year: July 1st
    second_half = np.argwhere(np.array(pinfo["CurrMonth"]) >= 7).ravel()
    pinfo["CurrYear"].iloc[second_half] += 1

    pinfo["age"] = pinfo["CurrYear"] - pinfo["DoB"]

    # create ground truth saliency maps for full sample
    gt_discrete = make_gt_smap(fixations.x, fixations.y)
    gt_continuous = make_gt_smap(fixations.x, fixations.y, blur=True)

    vis_smap(
        gt_continuous,
        save_path=n_fix_path / f"smap.png",
        cmap="binary_r",
        img_origin="upper",
    )

    vis_smap(
        gt_continuous, save_path=n_fix_path / "smap.pdf", img_origin="upper", show_img=False
    )

    # "ground truth saliency map", overlayed with free viewing image
    vis_smap(
        gt_continuous,
        overlay=free_viewing_gray,
        save_path=n_fix_path / "smap_overlayed.pdf",
        img_origin="upper",
        overlay_alpha=.4,
        show_img=False,
    )

    # show simple dataset statistics for demographics for filtered and unfiltered data
    def show_sumstats(
        df, description="", age_col="age", current_year=2023, gender_col="Gender",
    ):
        """Prints simple summary statistics"""
        gender_stats = df.groupby("Gender").size()
        gender_perc = round(gender_stats / gender_stats.sum() * 100, 2)
        print(
            f"----- {description}: ----- \n",
            f"N: {df.shape[0]}, \n",
            f"Age range: {df[age_col].min()}-{df[age_col].max()}, \n",
            f"Mean age: {round(df[age_col].mean(), 2)}, \n",
            f"SD age: {round(df[age_col].std(), 2)}, \n",
            f"Female (N, %): {gender_stats[0], gender_perc[0]}, \n",
            f"Male (N, %): {gender_stats[1], gender_perc[1]}, \n",
        )

    # keep only participant info and fixations with "valid demographics",
    pinfo["valid_demographics"] = (
        (pinfo["Valid"])
        & (pinfo["Valid Freeviewing"])
        & ((pinfo.DoB != 2000) & (pinfo.Gender != "OTHER"))
        & (pinfo.age >= 5)
        & (pinfo.age <= 59)
    )
    pinfo_valid_dem = pinfo[pinfo.valid_demographics]
    fix_valid_dem = fixations.loc[fixations.ID.isin(pinfo_valid_dem.ID)]

    if show:
        pinfo_filtered = pinfo.loc[pinfo.ID.isin(fixations.ID.unique())]
        show_sumstats(pinfo_filtered, "Full freeviewing sample")
        show_sumstats(pinfo_valid_dem, "Sample with valid demographics",)

    # make dataframe to include citations in results table
    abbr = [
        "AIM",
        "BMS",
        "CAS",
        "CVS",
        "DVA",
        "FES",
        "GBVS",
        "IKN",
        "IMSIG",
        "LDS",
        "QSS",
        "RARE2012",
        "SalGAN",
        "SSR",
        "SUN",
        "DeepGazeI",
        "DeepGazeII",
        "ICF",
        "DeepGazeIIE",
        "SALICON",
        "SAM",
        "Meaning map",
        "Central bias",
    ]
    citation = [
        "bruce2005saliency",
        "zhang2013saliency",
        "goferman2011context",
        "erdem2013visual",
        "hou2008dynamic",
        "rezazadegan2011fast",
        "GBVS",
        "itti1998model",
        "hou2011image",
        "fang2016learning",
        "schauerte2012quaternion",
        "riche2013rare2012",
        "pan2017salgan",
        "seo2009nonparametric",
        "zhang2008sun",
        "deepgazeI",
        "deepgazeII",
        "ICF",
        "deepgazeIIe",
        "jiang2015salicon",
        "SAM",
        "henderson2017meaning",
        "tatler2007central",
    ]
    citation = ["\\textcite{" + c + "}" for c in citation]
    mod_cite = pd.DataFrame({"Model": abbr, "Authors": citation})

    # calculate performance of baseline models
    # baseline model: all observers
    nss_all_observers = NSS(gt_continuous, gt_discrete)

    # baseline model: single observer
    # load single observer values if they were previously calculated and saved,
    # otherwise calculate single observer values from scratch
    oneobserver_nss_path = n_fix_path / "onehuman_nss.npy"
    if oneobserver_nss_path.exists():
        nss_single_observer = np.load(oneobserver_nss_path).mean()
    else:
        oneobserver_nss = Parallel(n_jobs=6)(
            delayed(oneobserver_perf)(i, fixations, "NSS")
            for i in tqdm(fixations.ID.unique())
        )
        oneobserver_nss = np.array(oneobserver_nss)
        np.save(oneobserver_nss_path, oneobserver_nss)
        nss_single_observer = oneobserver_nss.mean()

    # add all baselines into one DataFrame to combine with saliency maps results below
    base_models_nss = pd.DataFrame(
        {
            "Model": [
                "Fixation map",
                "Single observer",
            ],
            "NSS": [
                nss_all_observers,
                nss_single_observer,
            ],
            "Authors": ["" for x in range(2)],
        }
    )

    # compute NSS for all predicted saliency maps in a directory
    nss_results = evaluate_smaps_in_dir(smaps_path, gt_discrete, "NSS")

    # convert results to DataFrame, then add baseline models (and citations)
    nss_results = pd.DataFrame(nss_results).sort_values("NSS", ascending=False)
    nss_results = pd.merge(nss_results, mod_cite, on="Model")
    nss_results = pd.concat([base_models_nss, nss_results], axis=0)

    nss_central_bias = nss_results.loc[nss_results['Model'] == 'Central bias']
    nss_central_bias = nss_central_bias['NSS'].values[0]

    # add new column which reports relative performance compared to the all observers
    # and center prior baselines
    nss_results["Improvement"] = (
        (nss_results.NSS - nss_central_bias) / (nss_all_observers - nss_central_bias) * 100
    )
    nss_results["Improvement"] = [
        x + " %" for x in np.round(nss_results.Improvement, 2).astype("str")
    ]

    # save results and display if desired
    nss_results.to_csv(n_fix_path / "nss_results.csv", index=False)
    if show:
        print(nss_results.loc[:, ["Model", "NSS", "Improvement"]])

    # Next I calculate the difference in model performance based on sex. The following codeblock also creates some summary statistics for both subsamples.
    # ground truth fixation map for downsampled population with valid demographics
    tm_validdem = make_gt_smap(fix_valid_dem.x, fix_valid_dem.y)

    # compute NSS for all saliency maps in a directory, grouped by sex
    sex = pinfo_valid_dem.Gender.unique().tolist()
    sex.sort(reverse=True)
    results = {x: [] for x in (["Model", "full_sample", 'Mean'] + sex)}
    for f in smaps_path.iterdir():
        if ".png" in str(f):
            salmap = load_smap(f)

            # NSS results for full downsampled sample
            nss_valid = NSS(salmap, tm_validdem)
            results["Model"].append(f.stem)
            results["full_sample"].append(nss_valid)

            # NSS results split by sex
            nss_gender = []
            for s in sex:
                if s != 'OTHER':
                    # make subsample, create ground truth fixation map, calculate NSS
                    subsample_ID = pinfo_valid_dem.query(f"Gender == '{s}'").ID
                    subsample_fixations = fix_valid_dem[fix_valid_dem["ID"].isin(subsample_ID)]
                    tm_subsample = make_gt_smap(subsample_fixations.x, subsample_fixations.y)
                    nss = NSS(salmap, tm_subsample)
                    results[s].append(nss)
                    nss_gender.append(nss)
            
            results['Mean'].append(np.mean(nss_gender))

    # store results in DataFrame, then add column for difference between the male and
    # female samples
    nss_by_sex = pd.DataFrame(results)
    nss_by_sex = nss_by_sex.drop(['full_sample'], axis=1)

    nss_sex_melt = pd.melt(nss_by_sex, id_vars=['Model', 'Mean'], value_vars=['MALE', 'FEMALE'], var_name='Gender', value_name='NSS')
    nss_sex_melt['NSS deviation'] = nss_sex_melt['NSS'] - nss_sex_melt['Mean']
    nss_sex_melt['Percentage deviation'] = (nss_sex_melt['NSS deviation'] / nss_sex_melt['Mean']) * 100
    nss_sex_melt = nss_sex_melt.rename(columns={'Mean': 'Overall NSS'})
    nss_sex_melt['Gender'] = nss_sex_melt['Gender'].apply(lambda x: 'Male' if x == 'MALE' else 'Female')
    nss_sex_melt.to_csv(n_fix_path / f'gender_nss.csv')
    print(nss_sex_melt)

    # create summary statistics for the subsamples split by sex
    sumstats_by_sex = (
        pinfo_valid_dem.groupby("Gender")
        .aggregate(
            min=pd.NamedAgg(column="age", aggfunc="min"),
            max=pd.NamedAgg(column="age", aggfunc="max"),
            mean=pd.NamedAgg(column="age", aggfunc=np.mean),
            sd=pd.NamedAgg(column="age", aggfunc=np.std),
            n=pd.NamedAgg(column="age", aggfunc=np.count_nonzero),
        )
        .round(2)
        .reset_index()
    )

    # save results and display if desired
    sumstats_by_sex.to_csv(n_fix_path / f"sumstats_by_sex.csv", index=False)
    if show:
        print(sumstats_by_sex.round(3))

    distributions_per_sex_path = n_fix_path / "distributions_per_sex"
    distributions_per_sex_path.mkdir(exist_ok=True)

    for s in sex:
        # make subsample, create ground truth fixation map, calculate NSS
        subsample_ID = pinfo_valid_dem.query(f"Gender == '{s}'").ID
        subsample_fixations = fix_valid_dem[fix_valid_dem["ID"].isin(subsample_ID)]

        # save spatial distribution per age group
        spatial_distribution = make_gt_smap(
            subsample_fixations.x, subsample_fixations.y, blur=True
        )
        vis_smap(
            spatial_distribution,
            save_path=distributions_per_sex_path / f"{s}.pdf",
            show_img=False,
            img_origin="upper",
        )

    # for quickly viewing all distributions:
    vis_smaps_in_dir(distributions_per_sex_path, 
                    save_path=n_fix_path / f"all_smaps_gender.pdf",
                    ncols=1)

    # Following, potential performance differences of the visual saliency models over different age groups is investigated. First, I create age bins. Then I calculate the NSS over all bins, for all saliency maps. I then record the difference between the mean over all bins and each specific bin in order to have some measure of how different an age group is to the average performance over all age bins.
    # create age bins
    col = "age"
    conditions = [
        (pinfo_valid_dem[col] >= 6) & (pinfo_valid_dem[col] <= 11),
        (pinfo_valid_dem[col] >= 12) & (pinfo_valid_dem[col] <= 17),
        (pinfo_valid_dem[col] >= 18) & (pinfo_valid_dem[col] <= 23),
        (pinfo_valid_dem[col] >= 24) & (pinfo_valid_dem[col] <= 29),
        (pinfo_valid_dem[col] >= 30) & (pinfo_valid_dem[col] <= 35),
        (pinfo_valid_dem[col] >= 36) & (pinfo_valid_dem[col] <= 41),
        (pinfo_valid_dem[col] >= 42) & (pinfo_valid_dem[col] <= 47),
        (pinfo_valid_dem[col] >= 48) & (pinfo_valid_dem[col] <= 53),
        (pinfo_valid_dem[col] >= 54) & (pinfo_valid_dem[col] <= 59),
    ]
    names = ["06-11", "12-17", "18-23", "24-29", "30-35", "36-41", "42-47", "48-53", "54-59"]
    pinfo_valid_dem.loc[:,"age_bins"] = np.select(conditions, names, default=np.nan)
    pinfo_valid_dem.value_counts("age_bins").reset_index().sort_values("age_bins")
    pinfo_valid_dem = pinfo_valid_dem.loc[pinfo_valid_dem['age_bins'] != 'nan']

    # create summary statistics for the age bins
    sumstats_by_age = (
        pinfo_valid_dem.groupby("age_bins")
        .aggregate(
            minage=pd.NamedAgg(column="age", aggfunc="min"),
            maxage=pd.NamedAgg(column="age", aggfunc="max"),
            meanage=pd.NamedAgg(column="age", aggfunc=np.mean),
            n=pd.NamedAgg(column="age", aggfunc=np.count_nonzero),
            perc_male=pd.NamedAgg(
                column="Gender", aggfunc=lambda x: sum(x == "MALE") / np.count_nonzero(x)
            ),
            perc_female=pd.NamedAgg(
                column="Gender", aggfunc=lambda x: sum(x == "FEMALE") / np.count_nonzero(x)
            ),
        )
    )

    if show:
        print(sumstats_by_age.round(2))

    # compute NSS for all saliency maps in a directory, grouped by age bins
    # also save spatial distribution as plot per age group
    age_groups = sorted(pinfo_valid_dem["age_bins"].unique().tolist())

    ageresults = {x: [] for x in (["Model", "mean"] + age_groups)}

    for f in smaps_path.iterdir():
        if ".png" in str(f):
            salmap = load_smap(f)

            # NSS ageresults split by age
            nss_all_models = []
            for age_group in age_groups:
                # make subsample, create ground truth fixation map, calculate NSS
                subsample_ID = pinfo_valid_dem.query(f"age_bins == '{age_group}'").ID
                subsample_fixations = fix_valid_dem[fix_valid_dem["ID"].isin(subsample_ID)]
                tm_subsample = make_gt_smap(subsample_fixations.x, subsample_fixations.y)
                nss = NSS(salmap, tm_subsample)
                nss_all_models.append(nss)
                
            # calculate mean NSS over all age bins
            mean_nss_all_ages = np.array(nss_all_models).mean()
            ageresults["Model"].append(f.stem)
            ageresults["mean"].append(mean_nss_all_ages)

            # per age bin, calculate the difference from the mean and add to ageresults
            for nss, age_group in zip(nss_all_models, age_groups):
                ageresults[age_group].append(nss)

    # store ageresults in DataFrame
    nss_by_age = pd.DataFrame(ageresults).round(3)

    # Create a melted dataframe for plotting later
    nss_age_melt = pd.melt(nss_by_age, id_vars=['Model', 'mean'], value_vars=age_groups, var_name='Age', value_name='NSS')
    nss_age_melt['NSS deviation'] = nss_age_melt['NSS'] - nss_age_melt['mean']
    nss_age_melt['Percentage deviation'] = (nss_age_melt['NSS deviation'] / nss_age_melt['mean']) * 100
    nss_age_melt = nss_age_melt.rename(columns={'mean': 'Average NSS'})

    bins = []
    i = 0
    for age in range(len(age_groups)):
        for model in range(len(nss_by_age)):
            bins.append(i)
        i += 1

    nss_age_melt['Bin'] = bins
    nss_age_melt['Age'] = nss_age_melt['Age'].apply(lambda x: x.replace('06', '6'))
    print(nss_age_melt)
    nss_age_melt.to_csv(n_fix_path / f'age_nss.csv')

    # create aggregate columns which summarise performance per age bin compared to the
    # mean by counting how often the difference is positive or negative
    n_models = len(nss_by_age)
    count_positive = (
        (nss_by_age.select_dtypes("float") > 0)
        .sum()
        .reset_index()
        .set_index("index")
        .transpose()
    )
    count_negative = (
        (nss_by_age.select_dtypes("float") < 0)
        .sum()
        .reset_index()
        .set_index("index")
        .transpose()
    )
    balance = count_positive - count_negative

    # add aggregate columns to new DataFrame, then add them to previous NSS results
    counts = pd.concat([count_positive, count_negative, balance], axis=0)
    counts["Model"] = ["Count positive", "Count negative", "Difference"]
    counts["mean"] = ""
    counts = counts.loc[:, [list(counts)[-1]] + list(counts)[:-1]]
    nss_by_age = pd.concat([nss_by_age, counts])

    # adjust formatting for display
    nss_by_age.iloc[n_models:, 2:] = (
        nss_by_age.astype(str).iloc[n_models:, 2:].applymap(lambda x: x.replace(".0", ""))
    )

    # save ageresults and display if desired
    nss_by_age.to_csv(n_fix_path / f"nss_by_age.csv", index=False)
    if show:
        print(nss_by_age)

    distributions_per_age_path = n_fix_path / "distributions_per_age_bin"
    distributions_per_age_path.mkdir(exist_ok=True)

    for age_group in tqdm(age_groups):
        # make subsample, create ground truth fixation map, calculate NSS
        subsample_ID = pinfo_valid_dem.query(f"age_bins == '{age_group}'").ID
        subsample_fixations = fix_valid_dem[fix_valid_dem["ID"].isin(subsample_ID)]

        # save spatial distribution per age group
        spatial_distribution = make_gt_smap(
            subsample_fixations.x, subsample_fixations.y, blur=True
        )
        vis_smap(
            spatial_distribution,
            save_path=distributions_per_age_path / f"{age_group}.pdf",
            show_img=False,
            img_origin="upper",
        )

    # for quickly viewing all distributions:
    vis_smaps_in_dir(distributions_per_age_path,
                    save_path=n_fix_path / f"all_smaps_agebins.pdf",
                    ncols=2)

    # # make similarity maps for all saliency maps in a directory (saved as .png files)
    # # if they have not been created yet (i.e. if the amount of similarity maps does 
    # # not match the amount of saliency maps)
    sim_map_path = n_fix_path / "similarity_maps"
    sim_map_path.mkdir(exist_ok=True)

    if len(list(sim_map_path.glob("*.png"))) != len(list(smaps_path.glob("*.png"))):
        for f in smaps_path.iterdir():
            if not 'DS_Store' in f.name:
                smap = load_smap(f)
                output_path_diff = str(sim_map_path / f.name)
                vis_NSS_similarity(
                    smap, gt_discrete, save_path=output_path_diff, show_img=False, img_origin="lower"
                )
            
    # visualise all similarity maps
    vis_smaps_in_dir(
        sim_map_path,
        save_path=n_fix_path / f"all_similarity_maps.pdf",
        show_img=show,
    )

