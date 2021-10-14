import os
from postprocessing_analysis import * 

data_folder = os.path.join(os.getcwd(),'data/M0MDN3')
age = 'all'

def main():
    metric_names = ['slope', 'Backward distance traveled', 'Vel min']
    exp_folders = get_exp_folders([data_folder])

    cwa_line = [f for f in exp_folders if 'CWA' in f]
    ef_line = [f for f in exp_folders if 'EF' in f]
    if_line = [f for f in exp_folders if 'IF' in f]

    to_process = [cwa_line, ef_line, if_line]

    metrics = plot_metrics(to_process, age)
    
    for l, m in zip(metric_names,metrics):
        run_statistic_analysis(l, m)

if __name__ == "__main__":
    main()
