#!/usr/bin/env python2

import rospy
from std_msgs.msg import String

import pandas as pd
import numpy as np

import sys
sys.path.append('src/rpg_trajectory_evaluation/src/rpg_trajectory_evaluation/')

from align_trajectory import align_umeyama



# Aligns the dataframes
# df: dataframe to align
# returns: aligned dataframe
def align(df):
    
    df_aligned = pd.DataFrame(columns=['mission','x','y','z','t_x','t_y','t_z'])
    mission_list = df['mission'].unique()

    for mission in mission_list:
        df_mission = df.loc[df['mission'] == mission]
        df_mission = df_mission.reset_index(drop=True)

        # Align trajectories for mission
        model = np.array(df_mission[['t_x','t_y','t_z']])
        data = np.array(df_mission[['s_x','s_y','s_z']])

        s, R, t = align_umeyama(model, data)

        # Apply transformation to data
        # model = s * R * data + t
        model_aligned = s * np.dot(data, R.T) + t
        #model_aligned = s * R * data[] + t

        df_mission_aligned = pd.DataFrame(np.append(model_aligned,model,axis=1),columns=['x','y','z','t_x','t_y','t_z'])
        df_mission_aligned['mission'] = mission

        df_aligned = pd.concat([df_aligned,df_mission_aligned] ,ignore_index=True, sort=False)

        
    #print(df_aligned)

    return df_aligned

def calculate_error(df_aligned, df_comp):
    # Calculates the error between the aligned dataframe and the ground truth dataframe
    # df_align: aligned dataframe
    # df_gt: ground truth dataframe
    # returns: error dataframe
    columns_to_round = ['t_x','t_y','t_z']
    df_aligned[columns_to_round] = df_aligned[columns_to_round].round(6)


    # Should probably do this mission by mission
    df_merged = pd.merge(df_aligned, df_comp,  how='inner', left_on=['t_x', 't_y', 't_z'], right_on=['t_x', 't_y', 't_z'],sort=False)

    #print(df_merged.head())
    #print(df_merged)

    df_error = pd.DataFrame(columns=['mission','error_x','error_y','error_z'])
    df_error['mission'] = df_merged['mission']
    df_error['error_x'] = abs(df_merged['x'] - df_merged['gt_x'])
    df_error['error_y'] = abs(df_merged['y'] - df_merged['gt_y'])
    df_error['error_z'] = abs(df_merged['z'] - df_merged['gt_z'])

    return df_error


def main():

    df_align = pd.read_csv('/home/michbaum/Projects/maplab/data/loopclosure/lc_edges.csv', sep=',',names=['mission','t_x','t_y','t_z','s_x', 's_y','s_z'])
    df_gt = pd.read_csv('/home/michbaum/Projects/optag_EH/data/20240111_123911/groundtruth.csv', sep=',', usecols=[u'xPosition',u'yPosition',u'zPosition'])
    df_gt.rename(columns={'xPosition':'gt_x','yPosition':'gt_y','zPosition':'gt_z'}, inplace=True)
    df_drifty = pd.read_csv('/home/michbaum/Projects/optag_EH/data/20240111_123911/traj_estimate.csv', sep=',',usecols=[u'xPosition',u'yPosition',u'zPosition'])
    df_drifty.rename(columns={'xPosition':'t_x','yPosition':'t_y','zPosition':'t_z'}, inplace=True)

    #df_gt = pd.read_csv('/home/michbaum/Projects/optag_EH/data/20240111_174643/groundtruth.csv', sep=',')
    # print(df_gt.head())
    # print(df_drifty.head())
    df_drifty = df_drifty.round(6)

    df_gt.reset_index(drop=True, inplace=True)
    df_drifty.reset_index(drop=True, inplace=True)

    assert len(df_gt) == len(df_drifty), "Dataframes are not of the same length"
    df_comp = pd.DataFrame(columns=['gt_x','gt_y','gt_z','t_x','t_y','t_z'])
    df_comp = pd.concat([df_gt,df_drifty], axis=1)

    #rint(df_comp.head())

    df_aligned = align(df_align)

    #print(df_aligned.head())
    df_error = calculate_error(df_aligned, df_comp)
    print(df_error)


if __name__ == '__main__':
    main()