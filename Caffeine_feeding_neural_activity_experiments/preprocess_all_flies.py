"""
This script allows to pre-process the neural and behavioural data from scratch, 
i.e. starting with the raw data: green.tif and red.tif
Alternatively, neural data processing can be started from the green_com_warped.tif
This means center of mass registration and non-affine motion correction using ofco
have already been performed. This saves a lot of time because motion correction is very slow.
"""
import os
from copy import deepcopy

from twoppp import load, utils
from twoppp.pipeline import PreProcessFly

from flies import fly_dirs, conditions
from preprocessing_parameters import params

## IF YOU WANT TO SKIP THE MOTION CORRECTION, SET USE_PRECOMPUTED_MOTIONCORRECTION=True:
## this will base all further analysis on the green_com_warped files
USE_PRECOMPUTED_MOTIONCORRECTION = True
if USE_PRECOMPUTED_MOTIONCORRECTION:
    params.use_warp = False
    params.use_com = False
    params.ref_frame = ""

if __name__ == "__main__":
    for i_fly, (fly_dir, condition) in enumerate(zip(fly_dirs, conditions)):
        params_copy = deepcopy(params)

        print("Starting preprocessing of fly \n" + fly_dir)
        preprocess = PreProcessFly(fly_dir=fly_dir, params=params_copy)
        preprocess.run_all_trials()
        # dataframe processing not required for analysis in paper,
        # but can be used for additional analysis
        # preprocess.get_dfs()
