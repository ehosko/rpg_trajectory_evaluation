#!/usr/bin/env python2

import rospy
from std_msgs.msg import String

import pandas as pd
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.append('src/rpg_trajectory_evaluation/src/rpg_trajectory_evaluation/')

from align_trajectory import align_umeyama

# Aligns the dataframes
# df: dataframe to align
# returns: aligned dataframe
def align(df,df_drifty, threshold=10):
    
    df_aligned = pd.DataFrame(columns=['mission','x','y','z','t_x','t_y','t_z'])
    mission_list = df['mission'].unique()


    print("Number of rows in dataframe: " + str(len(df)))
    print("Number of missions: " + str(len(mission_list)))

    count = 0
    for mission in mission_list:
        df_mission = df.loc[df['mission'] == mission]
        df_mission = df_mission.reset_index(drop=True)

        # Align trajectories for mission
        model = np.array(df_mission[['t_x','t_y','t_z']])
        data = np.array(df_mission[['s_x','s_y','s_z']])

        # model = s * R * data + t
        s, R, t = align_umeyama(data, model)

        # Extend model data by interpolating
        columns_to_round = ['t_x','t_y','t_z']
        df_mission[columns_to_round] = df_mission[columns_to_round].round(4)
        df_merged = pd.merge(df_drifty, df_mission,  how='left', left_on=['t_x', 't_y', 't_z'], right_on=['t_x', 't_y', 't_z'],sort=False)


        # Find the indices of the rows without NaN values
        non_nan_indices = df_merged.index[(df_merged.isna().sum(axis=1) == 0)]
        #print(non_nan_indices)

        # Find the indices of the rows between rows without NaN values
        between_indices = non_nan_indices.to_list()
        for i in range(len(non_nan_indices) - 1):
            start_index = non_nan_indices[i]
            end_index = non_nan_indices[i + 1]
            if(end_index - start_index < threshold):
                between_indices.extend(range(start_index + 1, end_index))

        # Keep only the rows between rows without NaN values
        df_merged = df_merged.loc[between_indices]
        df_merged = df_merged.sort_index()
        #df_merged.reset_index(drop=True, inplace=True)

        align_data = np.array(df_merged[['t_x','t_y','t_z']])

        model_aligned = s * np.dot(align_data, R.T) + t
        #model_aligned = s * R * data[] + t

        df_mission_aligned = pd.DataFrame(np.append(model_aligned,align_data,axis=1),columns=['x','y','z','t_x','t_y','t_z'])
        df_mission_aligned['mission'] = mission

        df_aligned = pd.concat([df_aligned,df_mission_aligned] ,ignore_index=True, sort=False)

        count += 1
        #print(count)
        #print("Number of rows in aligned dataframe: " + str(len(model_aligned)))

        
    #print(df_aligned)

    print("Number of rows in aligned dataframe: " + str(len(df_aligned)))
    return df_aligned

def calculate_error(df_aligned, df_comp):
    # Calculates the error between the aligned dataframe and the ground truth dataframe
    # df_align: aligned dataframe
    # df_gt: ground truth dataframe
    # returns: error dataframe
    columns_to_round = ['t_x','t_y','t_z']
    df_aligned[columns_to_round] = df_aligned[columns_to_round].round(4)


    # Should probably do this mission by mission
    df_merged = pd.merge(df_aligned, df_comp,  how='inner', left_on=['t_x', 't_y', 't_z'], right_on=['t_x', 't_y', 't_z'],sort=False)

    #print(df_merged.head())
    #print(df_merged)

    df_error = pd.DataFrame(columns=['mission','error'])
    df_error['mission'] = df_merged['mission']
    df_error['error'] = ((df_merged['x'] - df_merged['gt_x'])**2 + (df_merged['y'] - df_merged['gt_y'])**2 + (df_merged['z'] - df_merged['gt_z'])**2)**0.5
    # df_error['error_y'] = abs(df_merged['y'] - df_merged['gt_y'])
    # df_error['error_z'] = abs(df_merged['z'] - df_merged['gt_z'])

    #print(df_error.size)

    return df_error

def calculate_residual_error(df_aligned):
    # df_align: aligned dataframe
    # returns: error dataframe

    df_error = pd.DataFrame(columns=['mission','error'])
    df_error['mission'] = df_aligned['mission']
    df_error['error'] = ((df_aligned['x'] - df_aligned['t_x'])**2 + (df_aligned['y'] - df_aligned['t_y'])**2 + (df_aligned['z'] - df_aligned['t_z'])**2)**0.5
    # df_error['error_y'] = abs(df_merged['y'] - df_merged['gt_y'])
    # df_error['error_z'] = abs(df_merged['z'] - df_merged['gt_z'])

    #print(df_error.size)

    return df_error

def plot_histogram(df, planner):
    # Group by 'mission' and calculate the squared error for each axis
    # grouped_df = df.groupby(df['mission']).agg({
    #     'error_x': 'mean',
    #     'error_y': 'mean',
    #     'error_z': 'mean'
    # })

    mission_list = df['mission'].unique()

    print("Number of missions: " + str(len(mission_list)))
    grouped_df = df.groupby(df['mission']).mean()

    grouped_df.dropna(inplace=True)
    print(len(grouped_df))
    print(grouped_df.head())
   

    # Plot histogram
    plt.hist(grouped_df['error'], bins=np.linspace(0,7,14), edgecolor='black')
    plt.xlabel('Average Euclidean Distance [m]')
    plt.ylabel('Frequency')
    plt.title('Average Squared Error of '+ str(len(grouped_df))+ ' Missions for ' + planner + ' Planner')

    plt.savefig('/home/michbaum/Projects/maplab/data/loopclosure/maze/Plots/' + planner + 'Error.png')
    plt.show()

def plot_points(df, planner):

    plt.scatter(df['t_x'],df['t_y'], s=1,c='r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trajectory of ' + planner + ' Planner')

    plt.savefig('/home/michbaum/Projects/maplab/data/loopclosure/maze/Plots/' + planner + 'Trajectory.png')
    plt.show()

def main():

    # TODO (ehosko): Read from yaml file - define paths to csv files
    planner = 'reconstruction'

    df_align = pd.read_csv('/home/michbaum/Projects/maplab/data/loopclosure/maze/'+ planner +'_eval/lc_edges.csv', sep=',',names=['mission','t_x','t_y','t_z','s_x', 's_y','s_z'])
    df_gt = pd.read_csv('/home/michbaum/Projects/optag_EH/data/maze_'+ planner +'_eval/groundtruth.csv', sep=',', usecols=[u'xPosition',u'yPosition',u'zPosition'])
    df_gt.rename(columns={'xPosition':'gt_x','yPosition':'gt_y','zPosition':'gt_z'}, inplace=True)
    df_drifty = pd.read_csv('/home/michbaum/Projects/optag_EH/data/maze_'+ planner +'_eval/traj_estimate.csv', sep=',',usecols=[u'xPosition',u'yPosition',u'zPosition'])
    df_drifty.rename(columns={'xPosition':'t_x','yPosition':'t_y','zPosition':'t_z'}, inplace=True)

    #df_gt = pd.read_csv('/home/michbaum/Projects/optag_EH/data/20240111_174643/groundtruth.csv', sep=',')
    # print(df_gt.head())
    # print(df_drifty.head())
    df_drifty = df_drifty.round(4)

    df_gt.reset_index(drop=True, inplace=True)
    df_drifty.reset_index(drop=True, inplace=True)

    assert len(df_gt) == len(df_drifty), "Dataframes are not of the same length"
    df_comp = pd.DataFrame(columns=['gt_x','gt_y','gt_z','t_x','t_y','t_z'])
    df_comp = pd.concat([df_gt,df_drifty], axis=1)

    #print(df_comp.head())


    df_aligned = align(df_align, df_drifty)

    #print(df_aligned.head())
    df_error = calculate_error(df_aligned, df_comp)

    plot_histogram(df_error,planner)
    

    df_error = calculate_residual_error(df_aligned)
    plot_histogram(df_error,planner+'Residual')
    #print(df_error)

    plot_points(df_aligned, planner)


if __name__ == '__main__':
    main()