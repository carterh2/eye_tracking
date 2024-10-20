"""
nemo_eyetracking
Copyright (C) 2022 Utrecht University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from analysis import run_analysis

# load participant info and fixations data
# (if relative pathing doesn't work exchange for direct directory paths)
data_path = Path.cwd() / "data"
result_path = Path.cwd() / "results"
repo_path = Path.cwd().parent

NUMBER_OF_FIXATIONS = 18

pinfo = pd.read_csv(
    repo_path / 'results' / "participant_info.csv",
    usecols=["ID", "Gender", "DoB", "Valid", "Valid Freeviewing"],
)

fixations = pd.read_csv(
    repo_path / 'data' / "compiled_fixations.csv", usecols=["ID", "Order", "offset", "avg_x", "avg_y"]
)

# keep only fixations with valid free viewing information
valid_IDs = pinfo.ID[pinfo["Valid Freeviewing"]]
fixations = fixations.loc[fixations.ID.isin(valid_IDs)]

# find the average offset of the 0th fixation
fix0 = fixations.loc[fixations['Order'] == 0]
print(f'\n0th fixation descriptives (n = {len(fix0)}):')
print('mean offset =', np.nanmean(fix0['offset']).round(3), 'seconds')
print('std  offset =', np.nanstd(fix0['offset']).round(3), 'seconds')
print('mdn  offset =', np.nanmedian(fix0['offset']).round(3), 'seconds')

# find the average offset of the ...th fixation
fix0 = fixations.loc[fixations['Order'] == NUMBER_OF_FIXATIONS]
print(f'\n{NUMBER_OF_FIXATIONS}th fixation descriptives (n = {len(fix0)}):')
print('mean offset =', np.nanmean(fix0['offset']).round(3), 'seconds')
print('std  offset =', np.nanstd(fix0['offset']).round(3), 'seconds')
print('mdn  offset =', np.nanmedian(fix0['offset']).round(3), 'seconds')

# extract IDs with ... or more fixations
IDs_enough_fixations = (
    fixations.groupby("ID")
        .size()
        .reset_index()
        .rename(columns={0: "Count"})
        .query(f"`Count` >= {NUMBER_OF_FIXATIONS}")
        .ID.to_list()
)

# filter fixations to only include the ... fixations for everyone with at least ...
# fixations and exclude fixation number zero
fixations = fixations[fixations["ID"].isin(IDs_enough_fixations)]
fixations = fixations.loc[lambda row: (row["Order"] != 0) & (row["Order"] <= NUMBER_OF_FIXATIONS)]

# remove fixations that are out of bounds of the screen resolution
fixations = fixations.loc[
    lambda row: (row["avg_x"] >= 0)
                & (row["avg_y"] >= 0)
                & (row["avg_x"] <= 1920)
                & (row["avg_y"] <= 1080)
]

# save processed fixations to new .csv file
fixations = (
    fixations.loc[:, ["ID", "avg_x", "avg_y", "Order"]]
        .rename(columns={"avg_x": "x", "avg_y": "y"})
)

save_path = Path.cwd() / "data"
fixations.to_csv(save_path / "filtered_fixations.csv", index=False)

run_analysis(fixations)
