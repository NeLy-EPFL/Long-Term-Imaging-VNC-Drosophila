"""
This script generates a separate video for each fly.
Each video contains behavioural data and raw 2p data for multiple trials selected.
"""
import os.path
from tqdm import tqdm
import numpy as np
from twoppp import load, utils
from twoppp.plot.videos import make_multiple_video_raw_dff_beh

green_name = "green.tif"  # name of the green channel tif to be generated
red_name = "red.tif"  # name of the red channel tif to be generated
i_cam = 5  # which camera to use for the behavioural video
FULL_VIDEO = False  # whether to use the full video or selected frames
VIDEO_POSTFIX = "test"  # if you want wo append a postfix to all videos

def make_longterm_functional_video(flies, out_dir):

    for i_fly, fly in enumerate(flies):
        print(f"Preparing raw files for fly {fly}.")
        
        trial_dirs = utils.readlines_tolist(fly["trial_txt"])
        beh_trial_dirs = utils.readlines_tolist(fly["beh_trial_txt"])

        if len(fly["selected_trials"]):
            trial_dirs = [trial_dir for i_trial, trial_dir in enumerate(trial_dirs) if i_trial in fly["selected_trials"]]
            beh_trial_dirs = [trial_dir for i_trial, trial_dir in enumerate(beh_trial_dirs) if i_trial in fly["selected_trials"]]

        # initialise file structure
        trial_processed_dirs = [os.path.join(trial_dir, load.PROCESSED_FOLDER)
                                for trial_dir in trial_dirs]
        _ = [utils.makedirs_safe(trial_processed_dir)
             for trial_processed_dir in trial_processed_dirs]
        green_dirs = [os.path.join(trial_processed_dir, green_name)
                      for trial_processed_dir in trial_processed_dirs]
        red_dirs = [os.path.join(trial_processed_dir, red_name)
                    for trial_processed_dir in trial_processed_dirs]

        # convert raw to tiff if necessary
        _ = [load.convert_raw_to_tiff(trial_dir=trial_dir, green_dir=green_dir, red_dir=red_dir)
             for trial_dir, green_dir, red_dir in tqdm(zip(trial_dirs, green_dirs, red_dirs))]


        selected_frames = [np.arange(int(start*fly["f_s"]),
                                    int(start*fly["f_s"] + fly["video_length_s"]*fly["f_s"]))
                        for start in fly["start_times_s"]]

        print("Prepared all raw files. Will now create video.")
        make_multiple_video_raw_dff_beh(
            dffs=[None for _ in trial_dirs],
            trial_dirs=trial_dirs,
            out_dir=out_dir,
            video_name=fly["name"]+"_"+VIDEO_POSTFIX,
            beh_dirs=beh_trial_dirs,
            sync_dirs=trial_dirs,
            camera=i_cam,
            greens=green_dirs,
            reds=red_dirs,
            text=None,
            time=False,
            select_frames=selected_frames if not FULL_VIDEO else None
        )



def main():
    txt_file_dir = os.path.join(load.NAS2_DIR_JB, "longterm", "longterm_T1_videos")
    fly1 = {
        "name": "longterm_T1_0930_fly5",  # becomes prefix of video
        # "trial_names": ["1 dpi", "5 dpi", "10 dpi"],  # before: text written in video. Now: not used anymore
        "trial_txt": os.path.join(txt_file_dir, "fly_0930_trial_dirs.txt"),  # a txt file containing all the trial dirs for that fly
        "beh_trial_txt": os.path.join(txt_file_dir, "fly_0930_beh_trial_dirs.txt"),  # a txt file containing all the behavioural trial dirs for that fly
        "selected_trials": [0,1,4],  # which trials of the ones in trial_txt and beh_trial_txt to use
        "start_times_s": [85, 65, 60],  # which time to start the video at in seconds
        "video_length_s": 15,  # how long he video is supposed to be in seconds
        "f_s": 10.74  # sampling frequency of the 2p data in Hz
    }
    fly2 = {
        "name": "longterm_T1_0928_fly4",
        # "trial_names": ["1 dpi", "5 dpi"],  # , "8 dpi"],
        "trial_txt": os.path.join(txt_file_dir, "fly_0928_trial_dirs.txt"),
        "beh_trial_txt": os.path.join(txt_file_dir, "fly_0928_beh_trial_dirs.txt"),
        "selected_trials": [0,1,4],
        "start_times_s": [0, 0, 0],
        "video_length_s": 30,
        "f_s": 10.74
    }
    flies = [fly1]  # , fly2]

    make_longterm_functional_video(flies, out_dir=txt_file_dir)


if __name__ == "__main__":
    main()