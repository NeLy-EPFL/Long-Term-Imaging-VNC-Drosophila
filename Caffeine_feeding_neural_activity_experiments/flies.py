import os.path
from longterm import load
from longterm import utils
BASE_DIR = os.path.join(load.NAS2_DIR_LH, "_dataverse", 
                        "test_copy_revisions")
# BASE_DIR = os.path.join(load.NAS2_DIR_LH, "_dataverse", 
#                        "test_copy_JB")  # Caffeine_feeding_neural_activity_experiments

conditions = [
    "High_Caffeine_Fly1",
    "High_Caffeine_Fly2",
    "High_Caffeine_Fly3",
    "Low_Caffeine_Fly1",
    "Sucrose_fly_1"
]

fly_dirs = [os.path.join(BASE_DIR, condition) for condition in conditions]

class Trial():
    def __init__(self, fly, trial_dir, beh_trial_dir=None, sync_trial_dir=None):
        super().__init__()
        self.fly = fly
        self.dir = trial_dir
        self.beh_dir = beh_trial_dir if beh_trial_dir is not None else trial_dir
        self.sync_dir = sync_trial_dir if sync_trial_dir is not None else trial_dir
        self.processed_dir = os.path.join(self.dir, load.PROCESSED_FOLDER)
        self.green_denoised = os.path.join(self.processed_dir, "green_denoised.tif")  # used
        self.green_raw = os.path.join(self.processed_dir, "green_com_warped.tif")  # used
        self.red_raw = os.path.join(self.processed_dir, "red_com_warped.tif")  # used
        self.green_norm = os.path.join(self.processed_dir, "green_denoised_norm.tif")  # used
        self.dff = os.path.join(self.processed_dir, "dff_denoised.tif")  # used
        self.name = self.dir.split("/")[-1].replace("_", " ")  # used
        self.twop_df = os.path.join(self.processed_dir, "twop_df.pkl")  # used TODO: include in processing pipeline


class Fly():
    def __init__(self, i_fly):
        super().__init__()
        self.i_fly = i_fly
        self.dir = fly_dirs[self.i_fly]
        self.i_flyonday = None
        self.date = None
        self.processed_dir = os.path.join(self.dir, load.PROCESSED_FOLDER)

        trial_dirs = load.get_trials_from_fly(self.dir, startswith="cs", exclude="processed")[0]
        self.selected_trials = range(len(trial_dirs))
        self.condition = conditions[self.i_fly]
        self.paper_condition = self.get_paper_condition(self.condition)  # used
        
        self.trials = [Trial(self, trial_dirs[i_trial]) 
                       for i_trial in self.selected_trials]
        self.fs = 16  # used
        self.summary_dict = os.path.join(self.processed_dir, "compare_trials.pkl")  # used
        wave_mask_names = ["mask_top.tif", "mask_left.tif", "mask_right.tif",
                           "mask_bottom.tif", "mask_gf_left.tif", "mask_gf_right.tif"]
        self.wave_masks = [os.path.join(self.processed_dir, wave_mask) for wave_mask in wave_mask_names]  # used
        self.trials_mean_dff = os.path.join(self.processed_dir, "trials_mean_dff.pkl")  # used
        self.raw_std = os.path.join(self.processed_dir, "raw_std.tif")  # used
        self.mask_fine = os.path.join(self.processed_dir, "cc_mask_fiji.tif")  # used
        self.wave_details = os.path.join(self.processed_dir, "wave_details.pkl")  # used
        self.rest_maps = os.path.join(self.processed_dir, "rest_maps.pkl")  # used
        self.correction = os.path.join(self.processed_dir, "illumination_correction.pkl")  # used

    @staticmethod
    def get_paper_condition(condition):
        if "high caff" in condition or "High" in condition:
            return "high caffeine"
        elif "low caff" in condition or "Low" in condition:
            return "low caffeine"
        elif "sucr" in condition or "Sucr" in condition:
            return "sucrose"
        else:
            return ""
    @property
    def trial_dirs(self):
        return [trial.dir for trial in self.trials]

    @property
    def trial_processed_dirs(self):
        return [trial.processed_dir for trial in self.trials]

    @property
    def trial_names(self):
        return [trial.name for trial in self.trials]


high_caff_flies = [Fly(i_fly) for i_fly in [0, 1, 2]]
high_caff_main_fly = high_caff_flies[0]

low_caff_flies = [Fly(i_fly) for i_fly in [3]]
low_caff_main_fly = low_caff_flies[0]

sucr_flies = [Fly(i_fly) for i_fly in [4]]
sucr_main_fly = sucr_flies[0]

all_flies = high_caff_flies + low_caff_flies + sucr_flies
main_flies = [high_caff_main_fly, low_caff_main_fly, sucr_main_fly]
