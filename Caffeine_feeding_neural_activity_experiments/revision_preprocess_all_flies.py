"""
This script allows to pre-process the neural and behavioural data used for the revision from scratch, 
i.e. starting with the raw data: green.tif and red.tif
Alternatively, neural data processing can be started from the green_com_warped.tif
This means center of mass registration and non-affine motion correction using ofco
have already been performed. This saves a lot of time because motion correction is very slow.
"""
import os
from copy import deepcopy

from twoppp import load, utils
from twoppp.pipeline import PreProcessFly

from revision_flies import fly_dirs, conditions
from preprocessing_parameters import params

## IF YOU WANT TO SKIP THE MOTION CORRECTION, SET USE_PRECOMPUTED_MOTIONCORRECTION=True:
## this will base all further analysis on the green_com_warped files
USE_PRECOMPUTED_MOTIONCORRECTION = True
if USE_PRECOMPUTED_MOTIONCORRECTION:
    params.use_warp = False
    params.use_com = False
    params.ref_frame = ""

# denoising, dff and summary stats not required for those flies
params.use_denoise = False
params.use_dff = False
params.make_summary_stats = False

if __name__ == "__main__":
    for i_fly, (fly_dir, condition) in enumerate(zip(fly_dirs, conditions)):
        params_copy = deepcopy(params)

        print("Starting preprocessing of fly \n" + fly_dir)
        preprocess = PreProcessFly(fly_dir=fly_dir, params=params_copy)
        preprocess.run_all_trials()
        if condition == "High_Caffeine_Fly1":
            # data frames only required for supplementary figure with walk/rest data of one fly
            preprocess.get_dfs()
