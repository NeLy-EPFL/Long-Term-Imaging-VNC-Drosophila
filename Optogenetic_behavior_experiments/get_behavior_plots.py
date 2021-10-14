import os
from postprocessing_analysis import * 


data_folder = os.path.join(os.getcwd(),'data/M0MDN3')
metric_names = ['slope', 'Backward distance traveled', 'Vel min']
#center = "Center" 
#collision_tolerance = 0.3 #s
#off_period_keep = 30 #s  
#window = 21 #length of the savgol filter


def main():
    exp_folders = get_exp_folders([data_folder])

    cwa_line = [f for f in exp_folders if 'CWA' in f]
    ef_line = [f for f in exp_folders if 'EF' in f]
    if_line = [f for f in exp_folders if 'IF' in f]

    to_process = [cwa_line, ef_line, if_line]

    metrics = plot_metrics(to_process)
    
    for l, m in zip(metric_names,metrics):
        print(l)
        run_statistic_analysis(l, m)

if __name__ == "__main__":
    main()