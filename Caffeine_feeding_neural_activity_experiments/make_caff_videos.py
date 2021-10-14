"""
To create the videos including behaviour and delta F/F, 
you have to run the preprocessing first to get the DFF/F.
The wave video waas created with denoised data. 
If you did not run the denoising again, this script will run it with raw data,
which is why it might appear rougher than in the paper.
"""
import os
from datetime import datetime
from tqdm import tqdm
from glob import glob

import pickle
import numpy as np
from scipy.ndimage import gaussian_filter1d, gaussian_filter

from twoppp import utils, load, rois
from twoppp.plot.videos import make_multiple_video_raw_dff_beh

from flies import high_caff_flies, high_caff_main_fly, low_caff_main_fly, sucr_main_fly

FILE_PATH = os.path.realpath(__file__)
CAFF_PATH, _ = os.path.split(FILE_PATH)
OUTPUT_PATH = os.path.join(CAFF_PATH, "outputs")
utils.makedirs_safe(OUTPUT_PATH)

def make_wave_video(flies, trials, output_dir, video_name, t_range=[-15, 15]):
    all_selected_frames = []
    greens = []
    reds = []
    trial_dirs = []
    beh_dirs = []
    sync_dirs = []
    texts = []
    masks = []
    norm_greens = []
    i_fly = 0
    old_i_fly = -1
    for fly, trial in tqdm(zip(flies, trials)):
        if fly.i_fly != old_i_fly:
            i_fly += 1
            old_i_fly = fly.i_fly
        wave_details_file = fly.wave_details[:-4] + f"_{trial}.pkl"
        if not os.path.isfile(wave_details_file):
            print("Could not find file: ", wave_details_file)
            print("Run the plotting script first!")
            continue
        with open(wave_details_file, "rb") as f:
            wave_details = pickle.load(f)
        selected_frames = np.arange(wave_details["i_global_max"] + fly.fs*t_range[0],
                                    wave_details["i_global_max"] + fly.fs*t_range[1])
        all_selected_frames.append(selected_frames)
        greens.append(fly.trials[trial].green_raw)
        if os.path.isfile(fly.trials[trial].red_raw):
            reds.append(fly.trials[trial].red_raw)
        else:
            print("Motion corrected red channel missing: ", fly.trials[trial].red_raw)
            print("Will only show green channel in video.")
            reds.append(None)
        trial_dirs.append(fly.trials[trial].dir)
        beh_dirs.append(fly.trials[trial].beh_dir)
        sync_dirs.append(fly.trials[trial].sync_dir)
        texts.append(f"fly {i_fly}")
        mask = utils.get_stack(fly.mask_fine) > 0
        masks.append(mask)

        if os.path.isfile(fly.trials[trial].green_norm):
            print("Loading pre-computed normalised green data.")
            green_norm = utils.get_stack(fly.trials[trial].green_norm)
        else:
            if os.path.isfile(fly.trials[trial].green_denoised):
                green_denoised = utils.get_stack(fly.trials[trial].green_denoised)
            else:
                print("Could not find the denoised green fluorescence data. Will instead use the raw one:")
                print(fly.trials[trial].green_raw)
                print("Expect video of lower quality than shown in the paper")
                green_denoised = utils.get_stack(fly.trials[trial].green_raw)

            q_low = np.quantile(green_denoised, 0.005, axis=0)
            q_high = np.quantile(green_denoised, 0.995, axis=0)
            green_norm = np.clip((green_denoised-q_low) / (q_high-q_low), a_min=0, a_max=1)
            utils.save_stack(fly.trials[trial].green_norm, green_norm)
        norm_greens.append(green_norm)


    make_multiple_video_raw_dff_beh(dffs=norm_greens, trial_dirs=trial_dirs, out_dir=output_dir,
                                    video_name=video_name, beh_dirs=beh_dirs, sync_dirs=sync_dirs,
                                    camera=6, greens=greens, reds=reds, mask=masks, text=texts, text_loc="beh",
                                    select_frames=all_selected_frames, share_lim=False, time=False,
                                    vmin=0, vmax=1, colorbarlabel="")

def make_behaviour_video(fly, i_trials, output_dir, video_name, start_time=0, video_length=None, fs=16.2):
    if not isinstance(start_time, list) and not start_time:
        if video_length is None:
            selected_frames = None
    elif not isinstance(start_time, list):
        start_frame = int(start_time*fs)
        N_frames = int(video_length*fs)
        selected_frames = [np.arange(start_frame, start_frame+N_frames) for i_trial in i_trials]
    elif isinstance(start_time, list):
        if video_length is None:
            raise ValueError("please specify a video length when using a start time > 0")
        else:
            start_frames = [int(s*fs) for s in start_time]
            N_frames = int(video_length*fs)
            selected_frames = [np.arange(start_frame, start_frame+N_frames)
                               for start_frame in start_frames]
    make_multiple_video_raw_dff_beh(
        dffs=[fly.trials[i_trial].dff for i_trial in i_trials],
        trial_dirs=[fly.trials[i_trial].dir for i_trial in i_trials],
        out_dir=output_dir,
        video_name=video_name,
        beh_dirs=[fly.trials[i_trial].beh_dir for i_trial in i_trials],
        sync_dirs=[fly.trials[i_trial].sync_dir for i_trial in i_trials],
        camera=6,
        greens=[fly.trials[i_trial].green_raw for i_trial in i_trials],
        reds=[fly.trials[i_trial].red_raw for i_trial in i_trials],
        mask=utils.get_stack(fly.mask_fine)>0,
        share_mask=True,
        text=["before feeding", "during feeding", "right after feeding", "25 min after feeding"],
        text_loc="beh",
        share_lim=not "high" in fly.paper_condition,
        vmax=600 if "sucr" in fly.paper_condition else None,
        time=False,
        downsample=2,
        colorbarlabel="dff",
        select_frames=selected_frames
        )

def main():

    extend_highcaff_flies = high_caff_flies[0:1]*2 + high_caff_flies[1:2]*2 + high_caff_flies[2:3]
    wave_trials = [-2, -1, -2, -1, -2]
    
    make_wave_video(extend_highcaff_flies, trials=wave_trials, output_dir=OUTPUT_PATH, video_name="_supvid_waves")
    
    print("The following videos can only be created if the entire pre-processind pipeline was run again.")
    try:
        video_length = 30  # s
        start_times = [0, 41, 158, 128]
        make_behaviour_video(high_caff_main_fly, [0,2,3,-2], OUTPUT_PATH, video_name="_supvid_highcaff_short",
                            start_time=start_times, video_length=video_length)

        start_times = [30, 143, 163, 195]
        make_behaviour_video(low_caff_main_fly, [0,2,3,-1], OUTPUT_PATH, video_name="_supvid_lowcaff_short",
                            start_time=start_times, video_length=video_length)
        
        start_times = [101, 142, 141, 52]
        make_behaviour_video(sucr_main_fly, [0,2,3,-1], OUTPUT_PATH, video_name="_supvid_succrose_short",
                            start_time=start_times, video_length=video_length)
    except FileNotFoundError as e:
        print(e)
        print("You have to run the processing pipeline before making dff + behaviour videos.")



if __name__ == "__main__":
    main()