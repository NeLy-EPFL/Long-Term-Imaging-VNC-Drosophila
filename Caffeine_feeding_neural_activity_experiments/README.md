# Code to pre-process functional two-photon and behaviour data of the feeding recordings.

This package relies installing the twoppp package according to the instructions:
https://github.com/NeLy-EPFL/twoppp
Also see the environment.yml file for package versions used to produce the results.

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
Once you have downloaded the data, you can run the following Pyhon scripts to reproduce the results:

## Contents
1. ```make_caff_figures.py```: run this file to reproduce Figure 4 from the paper (caffeine feeding). If you have not re-run the pre-processing, this script will use raw in stead of denoised data. As a result, the plots will appear noisier.
2. ```make_caff_videos.py```: run this file to reproduce the supplementary videos showing the caffeine wave or behaviour before/during/after feeding. If you have not re-run the pre-processing, this script will use raw in stead of denoised data. As a result, the videos will much! noisier. Only the wave video can be re-generated without re-running the pre-processing.
3. ```preprocess_all_flies.py```: run this file to reproduce the data pre-processing from raw data. You have two options:
    1. run neural data processing from scrath using only the green.tif and red.tif
    2. don't re-do the motion correction and use the provided DeepInterpolation model. This saves a lot of time and uses the green_com_warped.tif, which is the output of the optical flow motion correction.


