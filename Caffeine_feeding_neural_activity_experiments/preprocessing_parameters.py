import sys
import os.path
PACKAGE_PATH = "/home/jbraun/projects/longterm"
sys.path.append(PACKAGE_PATH)

from longterm.pipeline import PreProcessParams
    
params = PreProcessParams()
params.genotype = "J1xM5"

params.breadth_first = True
params.overwrite = False

params.use_warp = True
params.use_denoise = True
params.use_dff = True
params.use_df3d = False
params.use_df3dPostProcess = False
params.use_behaviour_classifier = False
params.select_trials = False
params.cleanup_files = False
params.make_dff_videos = False
params.make_summary_stats = True
params.ball_tracking = "opflow"

params.green_denoised = "green_denoised.tif"
params.dff = "dff_denoised.tif"
params.dff_baseline = "dff_baseline_denoised.tif"
params.dff_mask = "dff_mask_denoised.tif"
params.dff_video_name = "dff_denoised" 
params.dff_beh_video_name = "dff_beh"

params.i_ref_trial = 0
params.i_ref_frame = 0
params.use_com = True
params.post_com_crop = True
params.post_com_crop_values = [64, 80]
params.save_motion_field = False
params.ofco_verbose = True
params.motion_field = "w.npy"

params.denoise_crop_size = (352, 576)
params.denoise_train_each_trial = False
params.denoise_train_trial = 0
params.denoise_correct_illumination_leftright = True
params.denoise_final_dir = "denoising_run_correct"
params.denoise_tmp_data_dir = os.path.join(os.path.expanduser("~"), "tmp", "deepinterpolation", "data")
params.denoise_tmp_run_dir = os.path.join(os.path.expanduser("~"), "tmp", "deepinterpolation", "runs")

# dff params
params.dff_common_baseline = True
params.dff_baseline_blur = 10
params.dff_baseline_med_filt = 1
params.dff_baseline_blur_pre = True
params.dff_baseline_mode = "convolve"
params.dff_baseline_length = 10
params.dff_use_crop = None
params.dff_manual_add_to_crop = 20
params.dff_blur = 0
params.dff_min_baseline = 0
params.dff_baseline_exclude_trials = None

params.dff_video_pmin = None
params.dff_video_vmin = 0
params.dff_video_pmax = 99
params.dff_video_vmax = None
params.dff_video_share_lim = True
params.default_video_camera = 6