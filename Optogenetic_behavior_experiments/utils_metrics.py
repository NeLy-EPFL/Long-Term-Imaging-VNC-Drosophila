import numpy as np
import seaborn as sns

def bootsamplingCI(full_pop, number_bootstrap, confidence):
    
    """
    Compute the bootsampling confidence interval
    """
    
    means = np.zeros(number_bootstrap)
    
    if np.isnan(full_pop).any() and not np.isnan(full_pop).all():
        full_pop = np.array(full_pop.values)[np.where(np.logical_not(np.isnan(full_pop)))]
        
    for i in range(number_bootstrap):
        sample = np.random.choice(full_pop, size=len(full_pop), replace=True)
        means[i] = np.nanmean(sample)
    
    sorted_means = np.sort(means)
    low_val = (1-confidence)/2
    up_val = 1-low_val
        
    return [sorted_means[int(np.floor(number_bootstrap*low_val))], sorted_means[int(np.ceil(number_bootstrap*up_val))]]

def get_mean_and_CI(full_pop, number_bootstrap, confidence):
    
    """
    Return both the mean and the confidence interval of the population
    full pop
    """
    
    down, up = bootsamplingCI(full_pop, number_bootstrap, confidence)
    
    return [np.nanmean(full_pop), down, up]

def test_sequentiality(l, item, n):
    
    """
    Test wether an array contains n times item in a row
    """

    if n>len(l): return False, len(l)
    s = 0
    first = 0
    for i in range(len(l)):
        if l[i] != item:
            s = 0
            first = i
        else:
            s = s+1
            if s == n: return True, first

    return False, len(l)

def get_frames_to_ignore(fly_dat, stimulations, off_period_keep, collision_tolerance, fps):
    
    """
    Get the frames to ignore either because in the off period they are out
    of the keep window or in the on period because the fly collided longer
    than the time collision_tolerance
    """
    
    frame_to_ign = []
    frame_off_keep = fps*off_period_keep
    
    for stim in stimulations:
        
        period_len = fly_dat[stim]["stopFrame"]-fly_dat[stim]["startFrame"]
        
        if "off" in stim:
            #only keep the values that are interesting (eg 3sec at the beginning and end of off period)
                
            if frame_off_keep*2 < period_len:
                #only done if time to keep shorter than the period
                    
                frame_to_ign.extend(np.arange(frame_off_keep, period_len - frame_off_keep) + fly_dat[stim]["startFrame"])
                              
        if "on" in stim:
        # Do not show data if prolonged collision
            collision = np.logical_or(fly_dat[stim]["wallCollisions"], fly_dat[stim]["flyCollisions"])
            collide, collision_id = test_sequentiality(collision, True, int(collision_tolerance*fps))
               
            if collide:
                frame_to_ign.extend(np.arange(collision_id, period_len) + fly_dat[stim]["startFrame"])
    
    return (frame_to_ign,)

def nice_violinplot(x, y, data, ax, hue=None, colors=None):
    
    """
    Plot nice violin plots for the distribution y
    Showing both the data points and the violin
    """
    
    alpha = 0.2
    if colors==None:
        axis = sns.violinplot(x=x, y=y, hue=hue, data=data, ax=ax, inner="quartile")
    else:
        axis = sns.violinplot(x=x, y=y, hue=hue, data=data, ax=ax, inner="quartile", palette=colors)

    for violin in axis.collections:
        violin.set_alpha(alpha)

    if colors==None:
        ax2 = sns.stripplot(x=x, y=y, hue=hue, data=data, jitter=True, dodge=True, ax=ax)
    else:
        ax2 = sns.stripplot(x=x, y=y, hue=hue, data=data, jitter=True, dodge=True, ax=ax, palette=colors)
        
    handles, labels = ax.get_legend_handles_labels()
    half_len_handle = len(handles)//2
    l = ax.legend(handles[0:half_len_handle], labels[0:half_len_handle], bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title =hue)
    #ax = sns.swarmplot(x='angle', y='norm_error', hue='leg', data=df_errors, zorder=0, size=3)
    
    return

def get_slope (line, time):
    
    """
    Compute the slope from the first frame of the stimulation until the
    minimal velocity (integrate both amplitude and delay)
    """
    
    try:                 
        ind = np.nanargmin(line)
    except:
        return float('nan')
    if ind == 0:
        return float('nan')
    else:
        return (line[ind]-line[0])/(time[ind]-time[0])
        
    
def get_AOC(line, time):
    
    """
    Compute the area between the velocity curve and under the y=0 axis
    Linked to the distance walked backward, integrate notion of translational
    speed and time spent at this speed
    """
    
    step = time[1]-time[0]
    under0line = line[np.where(line<0)]
    
    if np.isnan(line).all(): 
        return float("nan")
    
    return np.dot(under0line, step*np.ones_like(under0line))


def get_min_vel(line, time):
    
    """
    Return the minimal velocity observed and the time where it was obseved
    """
    
    return np.nanmin(line), time[np.argmin(line)]-time[0]
