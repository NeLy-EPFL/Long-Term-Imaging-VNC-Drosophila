# Code for reproducing the behavior experiments analysis.

These scripts compute the velocity from tracked flies and calculate the metrics for analyzing the velocity profiles.
The tracking data from all the behavioral experiments are in the ```data``` folder. IF folders correspond to intact flies, CWAF folders correspond to sham dissected flies, and EF folders correspond to implanted flies. 

## Requirements:
- Anaconda for Python 3.7

## Installation:

- Create a conda environment using the ```environment.yml``` file:
```bash
$ conda env create -f environment.yml
```
**(Installation time: 20 seconds)**

- Activate the environment
```bash
$ conda activate long_term
```

## Reproducing the behavioral analysis
- Run the script ```get_behavior_plots.py```: This script will generate the plots shown in Figure 2 c-d
    - If the variable ```age``` is changed from ```all``` to ```first```, ```second```, or ```third```, the same plots will be generated for the specific ages from the experiments, 1-3 dpi, 14-16 dpi, or 28-30 dpi, respectively, as shown in Figure S7.

- The plots will be saved in a results folder in a pdf format.
