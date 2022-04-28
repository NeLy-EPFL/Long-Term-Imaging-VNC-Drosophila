# Code to pre-process functional two-photon and behaviour data of the feeding recordings.

This package relies installing the twoppp package according to the instructions:
https://github.com/NeLy-EPFL/twoppp
Also see the environment.yml file for package versions used to produce the results.
The installation of the conda environment should take around 20-30 minutes on a "normal" desktop computer.
## Reproducing the results:
Download the data from the harvard dataverse.

Make sure to keep the same folder structure:
- fly
    - trial 1
        - processed
        - behData
        - ...
    - ...
    - trial X
    - processed
        - denoising_run_correct
        - ...
Once you have downloaded the data, you can run the following Pyhon scripts to reproduce the results.
In flies.py and in revision_flies.py, change the BASE_DIR variable to wherever you stored the downloaded data.

## Contents
1. ```make_caff_figures.py```: run this file to reproduce Figure 4 from the paper (caffeine feeding). Some of the plots are only included in (3.) If you have not re-run the pre-processing, this script will use raw instead of denoised data. As a result, the plots will appear noisier. If you run this script the first time, it will take around 10 min on a "normal" desktop computer. Subsequent runs will take less than a minute.
2. ```make_caff_videos.py```: run this file to reproduce the supplementary videos showing the caffeine wave or behaviour before/during/after feeding. If you have not re-run the pre-processing, this script will use raw in stead of denoised data. As a result, the videos will much! noisier. Only the wave video can be re-generated without re-running the pre-processing. This script takes around 5 minutes on a "normal" desktop computer if run after the make_caff_figures.py script, which saves some processing steps.
3. ```revision_make_caff_figures.py```: run this file to reproduce the statistical analysis included in Figure 4 of the paper and the supplementary plots for all 9 flies. If you run this script the first time, it will take around 10 min on a "normal" desktop computer. Subsequent runs will take less than a minute.
4. ```preprocess_all_flies.py```: run this file to reproduce the data pre-processing from raw data. You have two options:
    1. run neural data processing from scrath using only the green.tif and red.tif
    2. don't re-do the motion correction and use the provided DeepInterpolation model. This saves a lot of time and uses the green_com_warped.tif, which is the output of the optical flow motion correction.
    3. If you re-run the entire pre-processing from scratch, this script will run multiple weeks on a "normal" desktop computer. On a highly parallelised computing cluster, this can be cut down to ~1 day. If you use the provided pre-processed steps, for example the "green_com_warped.tif" or the already trained DeepInterpolation model, the preprocessing will be much faster and should finish in ~1 day depending on your computer.
5. ```revision_preprocess_all_flies.py```: same as (4.), but for additional flies used for statistical analysis during revision. This will only run basic pre-processing excluding denoising and dff computation because these processing steps were not used for additional flies.

