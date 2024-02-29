import os 
import pandas as pd
import numpy as np


def main():
    
    # enviroment = 'Warehouse'
    enviroment = 'maze'

    planner_list = ['drift_aware_floorplan', 'drift_aware', 'drift_aware_floorplan_TSP', 'drift_aware_TSP', 'reconstruction', 'exploration', 'example']
    number_runs = 5


    for planner in planner_list:
        print(planner)

        path = '/home/michbaum/Projects/optag_EH/data/'+ enviroment + '/' + planner + '_planner/'
        # path = f'/home/michbaum/Projects/optag_EH/data/{enviroment}/{planner}_planner/'

        if(planner == 'example'):
            path = '/home/michbaum/Projects/optag_EH/data/'+ enviroment + '/' + planner + '_config/'

        new_path = '/home/michbaum/Projects/trajectory_evaluation/src/rpg_trajectory_evaluation/results/optag/workstation/'

        # Get all subdirectories for each run
        for i in range(1, number_runs + 1):
            print(i)

            file_est = path + 'run_' + str(i) + '/traj_estimate.csv'
            file_gt = path + 'run_' + str(i) + '/groundtruth.csv'


            df_est = pd.read_csv(file_est)
            df_gt = pd.read_csv(file_gt)

            # df_est = df_est.iloc[::20, :]
            # df_gt = df_gt.iloc[::20, :]

            new_file_path = new_path + planner + '/workstation_' + planner + '_' + enviroment
            
            if not os.path.exists(new_file_path):
                os.makedirs(new_file_path)

            pd.DataFrame.to_csv(df_est, new_file_path  + '/stamped_traj_estimate' + str(i-1) + '.txt', index=False, sep=' ', header=False)
            pd.DataFrame.to_csv(df_gt, new_file_path + '/stamped_groundtruth' + str(i-1) + '.txt', index=False, sep=' ', header=False)
    

if __name__ == "__main__":
    main()