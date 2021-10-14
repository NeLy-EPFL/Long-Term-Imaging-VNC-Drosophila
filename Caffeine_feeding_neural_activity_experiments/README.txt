So far this package relies on manually installing the following package:
https://github.com/NeLy-EPFL/longterm/tree/develop
and adding the path to it in the python files.
the longterm package in turn relies on the following packages.
They should be installed in a python 3.7 anaconda environment in the following order:
- conda create -n longterm37 python=3.7
- conda activate longterm37
- conda install jupyter
- DeepFly3D: https://github.com/NeLy-EPFL/DeepFly3D
- deepinterpolation: https://github.com/NeLy-EPFL/deepinterpolation/tree/adapttoR57C10
- conda install pandas
- conda install numpy
- utils2p: https://github.com/NeLy-EPFL/utils2p
- ofco: https://github.com/NeLy-EPFL/ofco
- utils_video: https://github.com/NeLy-EPFL/utils_video
- df3dPostProcessing: https://github.com/NeLy-EPFL/df3dPostProcessing
- conda install os sys pathlib shutil numpy array gc copy tqdm sklearn pickle glob matplotlib math cv2 json pandas scipy


Also see the environment.yml file for package versions used to produce the results