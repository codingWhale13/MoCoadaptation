import csv
from colorsys import hls_to_rgb
import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.graph_objects as go

### CANNOT BE USED WITH OLD CSV FILES ###
### NEW VERSION ###

path='/home/oskar/Thesis/priori/state_space_comparison' # paths need to be correct
newline=''


def convert_key_to_tuple(key):
    #return a tuple of values based on which the keys are sorted
    key_values = key.split('_')
    key_values = [value for value in key_values if value] 
    return tuple(map(float, key_values))


def sort_dictionaries(path):
    #Set up dictionaries
    # value_sums = {}
    # value_sums_mean = {}
    # link_lengths = {}
    for directoryname in os.listdir(path):
        directorypath = os.path.join(path, directoryname)
        #Set up dictionaries
        # value_sums_mean[directoryname] = {}
        # value_sums[directoryname] = {}
        # link_lengths[directoryname] = {}
        if os.path.isdir(directorypath):
            for directoryname2 in os.listdir(directorypath):
                #Set new directory path
                directorypath2 = os.path.join(directorypath, directoryname2)
                directory_keyname = directoryname2.split('[')[-1:]
                directory_keyname = directory_keyname[0].replace("]", "_").replace(", ", "_")
                # Go through and save data to dictionaries
                if os.path.isdir(directorypath2):
                    # total_run_spd_reward = np.array([]) #reset when in new file
                    # total_energy_cons_reward = np.array([])
                    # link_lenghts_ind = np.array([])
                    for filename in os.listdir(directorypath2):
                        if filename.endswith(".csv"):
                            filepath = os.path.join(directorypath2, filename)
                            with open(filepath, newline=newline) as file:
                                reader = csv.reader(file)
                                rows = [] # list for read values
                                # total_run_spd_reward = np.array([]) #reset when in new directory
                                # total_energy_cons_reward = np.array([])
                                # link_lenghts_ind = np.array([])
                                for row in reader:
                                    print(row)
                                    rows.append(row)
                                print(rows)
                                # run_speed_sum_reward_sum = np.sum(np.array(rows[1], dtype=float))
                                # energy_cons_reward_sum = np.sum(np.array(rows[2], dtype=float))
                                # link_lenghts_ind = np.append(link_lenghts_ind, np.array(rows[0], dtype=float))
                                # total_run_spd_reward = np.append(total_run_spd_reward, run_speed_sum_reward_sum)
                                # total_energy_cons_reward = np.append(total_energy_cons_reward, energy_cons_reward_sum)
                                # #values to dict
                                # value_sums_mean[directoryname][directory_keyname] = {'running_speed_returns_sum_mean':np.mean(total_run_spd_reward), 'energy_consumption_returns_sum_mean':np.mean(total_energy_cons_reward)}
                                # link_lengths[directoryname][directory_keyname] = link_lenghts_ind
                                # value_sums[directoryname][directory_keyname] = {'running_speed_returns_sum': total_run_spd_reward , 'energy_consumption_returns_sum': total_energy_cons_reward}
    return value_sums, value_sums_mean, link_lengths


if __name__ == "__main__":
    
    #the main is becomming awfully long, maybe these should just be put to functions....
    
    value_sums, value_sums_mean, link_lengths = sort_dictionaries(path)
    # Sort dictionaries based on keys
    sorted_mean_value_sums = dict(sorted(value_sums_mean.items(), key=lambda item: convert_key_to_tuple(item[0])))
    sorted_value_sums = dict(sorted(value_sums.items(), key=lambda item: convert_key_to_tuple(item[0])))
    sorted_link_lengths = dict(sorted(link_lengths.items(), key=lambda item: convert_key_to_tuple(item[0]))) #Sort link lenghts
    labels_links = list(sorted_link_lengths.keys())

    # Sort the inner dictionaries based on keys
    # mean values
    for key, inner_dict in sorted_mean_value_sums.items():
        #print(inner_dict.items())
        sorted_mean_value_sums[key] = dict(sorted(inner_dict.items(), key=lambda item: convert_key_to_tuple(item[0])))
    # sums
    for key, inner_dict in sorted_value_sums.items():
        sorted_value_sums[key] = dict(sorted(inner_dict.items(), key=lambda item: convert_key_to_tuple(item[0]))) 
    # link lengths
    for key, inner_dict in sorted_link_lengths.items():
        sorted_link_lengths[key] = dict(sorted(inner_dict.items(), key=lambda item: convert_key_to_tuple(item[0]))) 


    #Calculate values to arrays
    reward_sums = np.array([(sorted_mean_value_sums[key1][key2]['running_speed_returns_sum_mean'], 
                                sorted_mean_value_sums[key1][key2]['energy_consumption_returns_sum_mean']) for key1 in sorted_mean_value_sums.keys() for key2 in sorted_mean_value_sums[key1].keys()])

    value_std = np.array([[np.std(sorted_value_sums[key1][key2]['running_speed_returns_sum'], axis=0),
                        np.std(sorted_value_sums[key1][key2]['energy_consumption_returns_sum'], axis=0)]
                        for key1 in sorted_mean_value_sums.keys() for key2 in sorted_mean_value_sums[key1].keys()])

    link_lengths_array = np.array([[weights[key] for key in weights] for weights in sorted_link_lengths.values()])
    link_lengths_mean_array = np.array([np.mean(list(weights.values()), axis=0) for weights in sorted_link_lengths.values()])
    link_lengths_std_array = np.array([np.std(list(weights.values()), axis=0) for weights in sorted_link_lengths.values()])