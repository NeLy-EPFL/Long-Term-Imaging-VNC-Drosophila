from utils_metrics import *
import numpy
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.signal import savgol_filter
from scipy import stats
import scikit_posthocs as sp


def get_exp_folders(parent_folder): 
    
    """
    return the folders and parameters from the yaml file
    """        
    dirs=[]
    for gen1 in parent_folder:
        lines = next(os.walk(gen1))[1]
        for gen2 in lines:
            if os.path.isdir(os.path.join(gen1, gen2)):
                dirs.append(os.path.join(gen1, gen2))
        
    if dirs==[]:
        raise Exception('No folders found.')
    
    return dirs

def compute_av_stim_frame(experiments_stim_frames): 
    
    """
    Compute the average frame accross the flies were te stimulation starts
    and end
    """
   
    av_stim_frame = {}
    
    for i, expe_stim_frame in enumerate(experiments_stim_frames):
        for stim in expe_stim_frame:
            if "on" in stim:
                if i == 0:
                    av_stim_frame[stim] = [expe_stim_frame[stim]]
                else:
                    av_stim_frame[stim].append(expe_stim_frame[stim])

    return av_stim_frame


def plot_velocity(line, stats, times, fpss, experiments_stim_frames, age, stimulation_type="p6-0", off_period_keep=30, stat_name="Translational velocities", unit = "[mm/s]", ZOOM=False, ConfInt=False, confidence=0.95, step_int_conf=2, create_fig=True, create_ax=True, color = "b"):
    
        """
        The function plot the statistic given in the parameter vs  time. 
        It shows the average trace and all taces vs time.
        The ZOOM parameter enable to display zooms on the stimulation periods
        The ConfInt parameter enable the display of confidence interval
        (computed on bootstrap samples)
        
        Parameters:
        ----------
        stats: list
            The statistics for each fly to plot (lines are flies, columns are frames)
        
        times: list 
            The times point where the statistic was computed
        
        fpss: list
            The fps for each fly
        
        off_period_keep: float
            The period to keep after and before the stimulation in seconds
        
        experiments_stim_frames: dict
            Vector of dictionary containing all stimulation start and stop frames 
        
        stimulation_type: string
            Stimultion protocol to selct flies on can be p1-0 or p6-0
        
        line: int
            The fly line to study in self.parameter["lines"] (int)
        
        stat_name: string
            The name of the statistic ploted for nice graph
        
        unit: string
            Unit of the statistic for nice graphs 
        
        ZOOM: bool
            If True the plot is composed of subplots with the global view and
            a zoom on each stimulations 
        
        ConfInt: bool
            If true the plot shows the mean statistc and the confidence interval
            at level confidence of the mean (computed using bootstrapping) 
        
        confidence: float
            Confidence level of the confidence interval, 
        
        step_int_conf: int
            Since bootstraping can be long the CI is only computed each step_int_conf points
        
        (used mainly to plot multiple traces in the same figure)    
        fig: matplotlib figure
            The figure to  plot the statistic on if False it is generated in the function
        
        ax: matplotlib axis
            The axis to  plot the statistic on if False it is generated in the function
        
        color: matplotlib color
            The color of the statistic trace
        
        Returns:
        plot_name: string
            Name of the plot (used mainly to plot multiple traces in the same figure)  
        
        Plots the time serie and in case of fig=False save the figure in the 
        figure folder specified in the yaml file
            
        """
        if not fpss:
            return

        try:
            frame_off_keep = off_period_keep*fpss[0]
            mean_vel = np.nanmean(stats, axis=0)
            nb_frames = np.shape(times)[1]
        
            av_stim_frame = compute_av_stim_frame(experiments_stim_frames)
            flag = False
            line_split = line.split('/')
            line_name = line_split[-2] + '-' + line_split[-1]
        except Exception as e:
            print(e)
            return
                
        if ZOOM:

            n_plots = len(av_stim_frame)+1
            
            if create_fig and create_ax:
                fig, ax = plt.subplots(nrows=n_plots, ncols=1, figsize=(14,7*n_plots))
                flag = True
                ax[0].set_title("{} of flies from strain {} stimulation {} as a function of time \n".format(stat_name, line_name, stimulation_type), fontsize=22)
            
            ax[0].plot(times[0], mean_vel, linewidth=2, color = color, label=line_name+" "+stimulation_type)
            ax[0].plot(np.transpose(times), np.transpose(stats), alpha = 0.25, color = "gray")
            ax[0].set_xlabel("Time [s]") 
            ax[0].set_ylabel("{} [{}]".format(stat_name, unit)) 
            ax[0].set_ylim(-20,20)

            av_stim_frame_mod={}
            for stim in av_stim_frame:
                av_stim_frame_mod[stim] = np.round(np.mean(av_stim_frame[stim], axis=0))
                ax[0].axvspan(times[0][int(av_stim_frame_mod[stim][0])], times[0][int(av_stim_frame_mod[stim][1])], alpha=0.1, color='red')
            ax[0].grid()
            
            for stim, ax in zip(av_stim_frame_mod, ax[1:]):
                
                start_stim = int(av_stim_frame_mod[stim][0])
                stop_stim = int(av_stim_frame_mod[stim][1])
                
                start_plot = start_stim-frame_off_keep
                if start_plot < 0:
                    start_plot = 0
                    
                stop_plot = stop_stim+frame_off_keep
                if stop_plot > len(times[0]):
                    stop_plot = -1
                
                ax.axvspan(times[0][start_stim], times[0][stop_stim], alpha=0.1, color='red', label=stim)
                ax.plot(times[0][start_plot:stop_plot], mean_vel[start_plot:stop_plot], linewidth=2, color = color, label=line_name)
                ax.plot(np.transpose(times)[start_plot:stop_plot], np.transpose(stats)[start_plot:stop_plot], color = "gray", alpha = 0.25)
                if flag:
                    ax.set_title("Zoom on stimulation period {}".format(stim))
                ax.set_xlabel("Time [s]") 
                ax.set_ylabel("{} [{}]".format(stat_name, unit))  
                ax.set_ylim(-20,20)
                ax.set_xlim((start_stim-frame_off_keep)/fpss[0], (stop_stim+frame_off_keep)/fpss[0])
                ax.grid()
            
            #plot_name = "{}_{}_plot_{}.png".format(line_name, stimulation_type, stat_name)
            plot_name = "{}_{}_plot_{}.eps".format(line_name, stimulation_type, stat_name)    
            
            if flag:
                folder_name = os.path.join("results", 'Velocities_traces')
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                fig.savefig(os.path.join(folder_name, plot_name), dpi=300, bbox_inches='tight')
                plt.close()
                flag = False
                
        if ConfInt:
            if create_fig and create_ax:
                fig, ax = plt.subplots(figsize=(14,7))
                flag = True
                
            n_flies = len(stats)
            
            stats_t = np.array(stats).T
            
            conf_int = []
            
            for i in range(0, len(stats_t), step_int_conf):
                ind_nan = np.isnan(stats_t[i])
                
                if all(ind_nan):
                    conf_int.append([float("nan"), float("nan")])
                else:
                    conf_int.append(bootsamplingCI(stats_t[i][~ind_nan], 1000, confidence))
 
            ax.plot(times[0], mean_vel, linewidth=2, color = color, label=line_name+" "+stimulation_type)
            
            x_conf_int = times[0][0::step_int_conf]
            
            low_conf_int = np.transpose(conf_int)[0]
            up_conf_int = np.transpose(conf_int)[1]
            
            ax.fill_between(x_conf_int, low_conf_int, up_conf_int, alpha=0.25, color = "gray", label="Confidence interval of the mean at {}%".format(confidence*100))
           
            for stim in av_stim_frame:
                
                av_stim_frame[stim] = np.round(np.mean(av_stim_frame[stim], axis=0))
                start_stim = int(av_stim_frame[stim][0])
                stop_stim = int(av_stim_frame[stim][1])
                ax.axvspan(times[0][start_stim], times[0][stop_stim], alpha=0.1, color='red')

            if flag:
                ax.set_title("{} of flies from strain {} stimulation {} as a function of time \n".format(stat_name, line_name, stimulation_type), fontsize=22)
            ax.set_xlabel("Time [s]") 
            ax.set_ylabel("{} [{}]".format(stat_name, unit)) 
            plt.legend()
            plt.grid()
            
            
            plot_name = r'{}_Confidence_interval_{}_plot_{}_clean.png'.format(line_name, stat_name, stimulation_type)    
            if flag:
                folder_name = os.path.join("results", 'Velocities_CI')
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                fig.savefig(os.path.join(folder_name, plot_name), dpi=300, bbox_inches='tight')
                plt.close()
                flag = False
            
        if not ZOOM and not ConfInt:
            
            if create_fig and create_ax:
                fig, ax=plt.subplots(figsize=(14,7))
                flag = True
            
            plt.plot(times[0], mean_vel, linewidth=2, color = color, label=line_name+" "+stimulation_type)
            plt.plot(np.transpose(times), np.transpose(stats), alpha = 0.05, color = 'gray')
            if flag:
                ax.set_title("{} of flies from strain {} stimulation {} as a function of time  \n".format(stat_name, line_name, stimulation_type), fontsize=22)
            ax.set_xlabel("Time [s]") 
            ax.set_ylabel("{} [{}]".format(stat_name, unit)) 
            ax.set_ylim([-20,20])
                
            plt.grid()
            #print(av_stim_frame)
            for stim in av_stim_frame:
                av_stim_frame[stim] = np.round(np.mean(av_stim_frame[stim], axis=0))
                ax.axvspan(times[0][int(av_stim_frame[stim][0])], times[0][int(av_stim_frame[stim][1])], alpha=0.1, color='red')
                         
            plot_name = r'{}_Classical_traces_{}_plot_{}_{}.pdf'.format(line_name, stat_name, stimulation_type, age)    
            #plt.legend()
            
            if flag:
                folder_name = os.path.join("results", 'Velocities_raw')
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                fig.savefig(os.path.join(folder_name, plot_name), dpi=300, bbox_inches='tight')
                
                plt.close()

        return plot_name

def get_key_info(line, concat=False):
    velocities = []
    exp_names = []
    fly_gen = []
    times = []
    experiments_stim_frames = []
    fly_nbr = 0
    fpss = []
    experiments=[]
    if not concat:
        list_line = [line]
    else:
        list_line = line

    for l in list_line:
        #print(l)
        path_gen_dict = os.path.join(l,'new_genotype_dict.npy')
        #print('path_gen_dict', path_gen_dict)
        new_genotype_dict = np.load(path_gen_dict, allow_pickle=True).item()

        exp_data =[[key, exp] for i, (key, exp) in enumerate(new_genotype_dict.items())] 
        experiments.extend(exp_data)
        genotype = '-'.join([l.split('/')[-2],l.split('/')[-1]])
        fly_gen.extend([genotype]*len(exp_data))

    #for a,b in experiments:
    #    print(b)
    save_gen = []
    first_rec = []
    second_rec = []
    third_rec = []

    max_len=0
    count_exp = 0
    prev_exp = ""
    for exp_i, [exp_key, experiment] in enumerate(experiments) :
        #print("exp_i ",exp_i)
        #print("exp key ", exp_key)
        #print("fly_gen[exp_i] ",fly_gen[exp_i])

        if fly_gen[exp_i] in save_gen:
            #print('in saved gen already')
            continue

        save_gen.append(fly_gen[exp_i])
        temp_date = []
        temp_ind = []
        temp_date.append(int(exp_key.split('_')[0]))
        temp_ind.append(0)
        try:
            for m in range(1,4):
                if fly_gen[exp_i] == fly_gen[exp_i+m]:
                    temp_date.append(int(experiments[exp_i+m][0].split('_')[0]))
                    temp_ind.append(m)
                    pass
        except Exception as e:
            print('\nException:' + str(e))
            pass

        temp_date_arr = np.array(temp_date)
        sort_index = np.argsort(temp_date_arr)
        if len(sort_index)==3:
            first_rec.append(experiments[exp_i+temp_ind[sort_index[0]]][0])
            second_rec.append(experiments[exp_i+temp_ind[sort_index[1]]][0])
            third_rec.append(experiments[exp_i+temp_ind[sort_index[2]]][0])
        if len(sort_index)==2:
            first_rec.append(experiments[exp_i+temp_ind[sort_index[0]]][0])
            second_rec.append(experiments[exp_i+temp_ind[sort_index[1]]][0])
        if len(sort_index)==1:
            first_rec.append(experiments[exp_i+temp_ind[sort_index[0]]][0])
    #print("first rec")  
    #print(first_rec)
    #print("second rec", second_rec)
    #print("third rec", third_rec)
    return first_rec, second_rec, third_rec


def get_velocities(first, second, third, line, age, off_period_keep=30, collision_tolerance=0.3, center="Center", window=21, stimulation_type="p6-0", savgol=True, concat=False, save_data=False, get_experiments_order=True):

    """
    This function compute the translational velocities of all flies with the ame stimulation protocol using the point provided in center.

    Parameters: 
    ----------
    line: list(int)
        The flies' line to study in self.parameter["lines"]

    off_period_keep: float
        The period to keep after and before the stimulation in seconds

    collision_tolerance: float
        If the fly collide (with the wall or another fly) for less than tol the datas are used

    center: string
        If centerSingle the fly centroid is the one of the classical tracking
        If centerMulti the fly centroid will be taken from deep lab cut

    window: int
        Length of the mooving average filter or savitzky golay filter
        on the positions

    stimulation_type: string
        Stimultion protocol to selct flies on can be p1-0 or p6-0

    ignore_odd_flies: bool
        If true the odd flies (orientation flipping during the experiment)
        listed in the odd_flies.csv file are excluded

    custom: bool
        If true the user can manually enter the fies he wants to compute
        the velocities 

    savgol: bool
        If True the velocity is computed using the savitzky golay algorithm
        If False the velocity is computed using forward Euler

    Returns:
    ----------
    velocities : list
        velocity of all flies

    times: list
        according times

    experiments_stim_frames: list(dict)
        vector of dictionary containing all stimulation start and stop frames 

    fpss: list
        fps of each fly

    WARNING when using the savitzky golay the colliding frames are kept 
    for the velocity calculation thus the interpolated line is compted on 
    some "invalid frames"
    When using forward euler and a moving average, the moving average window 
    use nan mean thus ignoring NaN ("invalid frames") the averages are always
    computed accross valid frames

    """  
    velocities = []
    exp_names = []
    fly_gen = []
    times = []
    experiments_stim_frames = []
    fly_nbr = 0
    fpss = []
    experiments=[]
    ratio = 32/832
    
    #print('line ', line)
    if not concat:
        list_line = [line]
    else:
        list_line = line

    for l in list_line:
        #print(l)
        path_gen_dict = os.path.join(l,'new_genotype_dict.npy')
        #print('path gen dict', path_gen_dict)
        new_genotype_dict = np.load(path_gen_dict, allow_pickle=True).item()

        exp_data =[[key, exp] for i, (key, exp) in enumerate(new_genotype_dict.items())] 
        experiments.extend(exp_data)
        genotype = '-'.join([l.split('/')[-2],l.split('/')[-1]])
        fly_gen.extend([genotype]*len(exp_data))

    max_len=0
    count_exp = 0
    prev_exp = ""

    if age is not "all":
        if age == "first":
            age_to_process = first
        if age == "second":
            age_to_process = second
        if age == "third":
            age_to_process = third
    
    for exp_i, [exp_key, experiment] in enumerate(experiments) :
        if age is not "all":
            if exp_key not in age_to_process:
                print("Skipping: experiment from a different age")
                continue

        try:
            if stimulation_type in exp_key:                
                fps = experiment["fps"]
                fpss.append(fps)
                stimulations = experiment["stimulation_paradigm"]
                flies = [key for key in experiment.keys() if "fly" in key]
                vel = []                    
                for fly_i, fly in enumerate(flies):
                    fly_nbr += 1
                    flyn_x = []
                    flyn_y = []
                    flyn_theta = []
                    stim_frame = {}

                    print("CHECK")

                    for stim in stimulations:
                        flyn_x.extend(experiment[fly][stim]['trajectories'][center]["x"])
                        flyn_y.extend(experiment[fly][stim]['trajectories'][center]["y"])
                        flyn_theta.extend(experiment[fly][stim]['trajectories'][center]["orientation"])
                        stim_frame[stim] = [experiment[fly][stim]["startFrame"], experiment[fly][stim]["stopFrame"]]

                    flyn_x = np.array(flyn_x)
                    flyn_y = np.array(flyn_y)        
                    flyn_theta = np.array(flyn_theta)

                    frames_to_ign = get_frames_to_ignore(experiment[fly], stimulations, off_period_keep, collision_tolerance, fps)

                    if savgol:

                        order = 2
                        dx = savgol_filter(flyn_x, window, polyorder=order, deriv=1, delta=1/fps)
                        dx[frames_to_ign] = float("nan")
                        dy = -savgol_filter(flyn_y, window, polyorder=order, deriv=1, delta=1/fps)
                        dy[frames_to_ign] = float("nan")
                        raw_speed = np.sqrt(dx**2 + dy**2)*ratio #velocity in mm/s 

                    else:

                        frames_to_ign = ([ind-1 for ind in frames_to_ign[0]],)

                        flyn_x[frames_to_ign] = float("nan")
                        flyn_y[frames_to_ign] = float("nan")
                        flyn_theta[frames_to_ign] = float("nan")

                        flyn_x = moving_avg(flyn_x, filter_len=window)
                        flyn_y = moving_avg(flyn_y, filter_len=window)       
                        dy = -np.diff(flyn_y)
                        dx = np.diff(flyn_x)
                        raw_speed = np.sqrt(dx**2 + dy**2)*ratio*fps



                    flyn_theta_rescaled = np.asarray([((360-(val-90))%360)*np.pi/180 for val in flyn_theta])

                    vtheta = np.arctan2(dy, dx)

                    if savgol:
                        proj = np.cos(flyn_theta_rescaled-vtheta)
                    else:
                        proj = np.cos(flyn_theta_rescaled[1:]-vtheta)


                    vel = np.multiply(raw_speed, proj) 

                    if velocities:

                        if max_len < len(vel):
                            max_len = len(vel)

                            for ind, v in enumerate(velocities):
                                if len(v)<max_len:
                                    v = np.pad(v, (0,max_len-len(v)), mode='constant', constant_values=float("nan")) 
                                    velocities[ind] = v
                                    times[ind] = np.arange(len(velocities[ind]))/fps
                        if max_len > len(vel):
                            vel = np.pad(vel, (0,max_len-len(vel)), mode='constant', constant_values=float("nan"))
                    else:
                        max_len = len(vel)



                    time = np.arange(len(vel))/fps

                    velocities.append(vel)
                    times.append(time)
                    experiments_stim_frames.append(stim_frame)
                    exp_names.append("_".join([fly_gen[exp_i],exp_key]))
        except Exception as e:
            print('\nException:' + str(e))
            pass

    if save_data:
        vel_data = {}
        vel_data['velocities'] = velocities
        vel_data['times'] = times
        vel_data['stim_frames'] = experiments_stim_frames
        vel_data['fps'] = fpss
        vel_path = os.path.join('results',line.split('/')[-1])
        if not os.path.exists(vel_path):
            os.makedirs(vel_path)
        np.save(vel_path+'/velocities_data.npy', vel_data, allow_pickle=True)

    
    return velocities, times, experiments_stim_frames, fpss, exp_names
    

def plot_metrics(lines, age = "all", stimulation_type = "p6-0", agg = "Line", bp=True, concat=True):

    """
    Save the velocity based metrics of all flies and display their 
    distribution
    The metrics are the folllowing:
    - slope until the first negative velocities
    - Area under the velocity curve (~backward distance)
    - Area over the velocity curve (~forward distance)
    - Minimal translational velocity
    - Maximal duration of the negative velocity peak
    The mean metric and its confidence interval is computed and saved 
    in the file

    Parameters:
    ----------
    lines_to_compute: list(string)
        The lines you want to compute the velocity metrics on 
        If "ALL" the metrics are computed for all lines

    stimulation_type: string
        Stimultion protocol to selct flies on can be p1-0 or p6-0

    agg: string
        Define what to aggregate the flies metric on.
        If "LinPer" the values are aggregated for the same stimulation period
        and line
        If "Line" the metreics are averaged over the whole fly line

    bp: bool
        If True the distribution of the fly metrics is plotted


    Returns:
    ----------
    0

    But plots the metrics distribution and save the metrics dependant upon 
    the parameters in the output file (data)
    """      

    my_indexes = pd.MultiIndex(levels=[[],[],[]], codes=[[],[],[]], names=["Fly_index", "Fly_line", "Stimulation"])
    my_columns = ["Experiment","Slope[mm/s\u00b2]", "Backward distance traveled [mm]", "Vmin [mm/s]"]
    metrics = pd.DataFrame(index=my_indexes, columns=my_columns, dtype=float)

    slope_all = []
    AOC_all = []
    vmin_all = []
    bout_all = []
    line_names=['CWA/all','EF/all','IF/all']
    
    for i, line in enumerate(lines):
        print(line)
        slope_line = []
        AOC_line = []
        vmin_line = []
        bout_line = []
        try:
            line_split = line.split('/')
        except:
            line_split = line[0].split('/')
        line_name = line_split[-2] + '-' + line_split[-1]
        first, second, third = get_key_info(line, concat=concat)

        velocities, times, experiments_stim_frames, fpss, exp_names = get_velocities(first, second, third, line, age, stimulation_type=stimulation_type, concat=concat)
        
        plot_velocity(line_names[i], velocities, times, fpss, experiments_stim_frames, age)        
        
        stim_on = []
        times = np.array(times)

        for key in experiments_stim_frames[0]:
            if "on" in key:
                stim_on.append(key)        
        for num, on in enumerate(stim_on):
            on_vel = []
            for i, [vel, on_frames, time] in enumerate(zip(velocities, experiments_stim_frames, times)):

                if on in on_frames.keys():
                    fly_on_vel = np.array(vel[on_frames[on][0]:on_frames[on][1]])
                    fly_on_time = np.array(time[on_frames[on][0]:on_frames[on][1]])
                else:
                    continue

                a = get_slope(fly_on_vel, fly_on_time)
                
                #AUC = get_AUC(fly_on_vel, fly_on_time)
                #print("The area under the curve is: {}".format(AUC))
                AOC = get_AOC(fly_on_vel, fly_on_time)
                #print("The area above the curve is: {}".format(AOC))
                vmin, delta_t_vmin = get_min_vel(fly_on_vel, fly_on_time)
                #print("The minimal velocity is {} attained in {} s ".format(vmin, delta_t_vmin))
                #dur, start = get_max_dur_neg_vel(fly_on_vel, fly_on_time)
                #print("Negative velocity is sustained for a maximal duration of {}s starting at sec {} in period {}".format(dur, start, on))

                data = [exp_names[i], a, AOC, vmin]
                index = (i, line_name, on)
                metrics.loc[index, :] = data

                slope_line.append(a)
                AOC_line.append(AOC)
                vmin_line.append(vmin)
                #bout_line.append(dur)
        slope_all.append(slope_line)
        AOC_all.append(AOC_line)
        vmin_all.append(vmin_line)
        #bout_all.append(bout_line)

    if not os.path.exists("results"):
        os.makedirs("results")
    if bp: 
        metric_cols = metrics.columns
        metric_cols = metric_cols[1:]
        print("METRICS COLS : ", metric_cols)
        fig, axs = plt.subplots(nrows=len(metric_cols), ncols=1, figsize=(14,7*len(metric_cols)))
    if agg == "Line":
        metrics.reset_index("Fly_line", inplace=True)
        #print(metrics.head())
        metrics_av = metrics.groupby("Fly_line").agg(lambda x : get_mean_and_CI(x, 1000, 0.95))
        #print(metrics_av)
        if bp: 
            for ax, col_name in zip(axs, metric_cols):
                #print("ax",ax,"col_name",col_name)
                nice_violinplot("Fly_line", col_name, metrics, ax)
            fig_name = f"{line_name}_lines_metrics_{stimulation_type}_{age}"
    if bp:
        fig.suptitle("Distribution of velocity metrics")
        fig.tight_layout(pad=3)

        if not os.path.exists("results"):
            os.makedirs("results")
        fig.savefig(os.path.join("results", fig_name+"_clean.pdf"),dpi=300)
        plt.close()
    metrics.to_csv(os.path.join("results", "Lines_Metrics_allexcept4threc{}.csv".format(stimulation_type)))

    return [slope_all, AOC_all, vmin_all]

def run_statistic_analysis(label, metric):

    metric = [[x for x in y if not np.isnan(x)] for y in metric]
    
    print(f"{label} kruskal ", stats.kruskal(metric[0],metric[1],metric[2],nan_policy="omit"))
    
    print(f"{label} post hoc conover", sp.posthoc_conover(np.array([metric[0],metric[1],metric[2]]),p_adjust='holm'))
    
