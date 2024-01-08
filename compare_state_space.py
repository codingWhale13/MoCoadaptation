import csv
from colorsys import hls_to_rgb
import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.graph_objects as go


##### USED FOR COMPARING STATES AND ACTION OF MODEL #####

path='/home/oskar/Thesis/priori/state_space_comparison' # paths need to be correct
newline=''


def convert_key_to_tuple(key):
    #return a tuple of values based on which the keys are sorted
    key_values = key.split('_')
    key_values = [value for value in key_values if value] 
    return tuple(map(float, key_values))



def sort_dictionaries(path):
    #Set up dictionaries
    state_values = {}
    action_values = {}
    for directoryname in os.listdir(path):
        directorypath = os.path.join(path, directoryname)
        #Set up dictionaries
        state_values[directoryname] = {}
        action_values[directoryname] = {}

        if os.path.isdir(directorypath):
            for directoryname2 in os.listdir(directorypath):
                #Set new directory path
                directorypath2 = os.path.join(directorypath, directoryname2)
                directory_keyname = directoryname2.split('[')[-1:]
                directory_keyname = directory_keyname[0].replace("]", "_").replace(", ", "_")
                # Go through and save data to dictionaries
                if os.path.isdir(directorypath2):
                    for filename in os.listdir(directorypath2):
                        if filename.endswith(".csv"):
                            filepath = os.path.join(directorypath2, filename)
                            with open(filepath, newline=newline) as file:
                                reader = csv.reader(file)
                                rows_states = [] # list for read values
                                rows_actions = []
                                for row in reader:
                                    row_values = []
                                    if 'States' in row:
                                        switch = False
                                        #print(switch)
                                        continue
                                    if 'Actions' in row:
                                        switch = True
                                        #print(switch)
                                        continue
                                    
                                    for index in row:
                                        row_values.append([float(value) for value in index.strip('[]').replace('\n', '').split()])
                                    if switch:
                                        rows_actions.append(row_values)
                                    else:
                                        rows_states.append(row_values)
                                        
                                states_n = np.array(rows_states, dtype=float)
                                action_n = np.array(rows_actions, dtype=float)
                                
                                # #values to dict
                                state_values[directoryname][directory_keyname] = {'states' : states_n}
                                action_values[directoryname][directory_keyname] = {'actions' : action_n}
                                
    return state_values, action_values


if __name__ == "__main__":
    
    state_values, action_values = sort_dictionaries(path) #value_sums, value_sums_mean, link_lengths = sort_dictionaries(path)
    
    #Sort dictionaries based on keys
    
    sorted_state_values = dict(sorted(state_values.items(), key=lambda item: convert_key_to_tuple(item[0])))
    sorted_action_values = dict(sorted(action_values.items(), key=lambda item: convert_key_to_tuple(item[0])))

    # Sort the inner dictionaries based on keys
    #states
    for key, inner_dict in sorted_state_values.items():
        sorted_state_values[key] = dict(sorted(inner_dict.items(), key=lambda item: convert_key_to_tuple(item[0])))
    # actions
    for key, inner_dict in sorted_action_values.items():
        sorted_action_values[key] = dict(sorted(inner_dict.items(), key=lambda item: convert_key_to_tuple(item[0]))) 

    #Calculate values to arrays
    states_model = np.array([sorted_state_values[key1][key2]['states'] for key1 in sorted_state_values.keys() for key2 in sorted_state_values[key1].keys()])
    actions_model = np.array([sorted_action_values[key1][key2]['actions'] for key1 in sorted_action_values.keys() for key2 in sorted_action_values[key1].keys()])

    print(states_model.shape)
    #print()
    print(actions_model.shape)
    
    
    #Extract relevant information
    
    weight_classes =  states_model.shape[0]
    num_iterations = states_model.shape[2]
    num_states = states_model.shape[3]

    # Create a figure and axis
    fig, ax = plt.subplots()
    colors = plt.cm.viridis(np.linspace(0, 1, weight_classes))
    iterations = 28 #-1 # switch the amount of iterations to plot more tests or not max 30 aka 29

    legend_added = set()
    legend_labels = []
    legend_handles = []
    
    unique_weight_groups = sorted(set(sorted_state_values.keys()), key=convert_key_to_tuple)
    #for j in range(weight_classes):
    for j, (key1, key2) in enumerate([(key1, key2) for key1 in sorted_state_values.keys() for key2 in sorted_state_values[key1].keys()]):
        mask = [key_compare == key1 for key_compare in sorted_state_values.keys()]
        weight_group = unique_weight_groups[np.where(mask)[0][0]] 
        for i in range(num_iterations):
            # for only certain iteration
            if i > iterations: #-1:
                if weight_group not in legend_added:
                    x_values = np.arange(i * 1001, (i + 1) * 1001)
                    y_values = states_model[j, :, i, :]  # Extract states for the current iteration
                    line = ax.plot(x_values, y_values, color=colors[j], alpha=0.5)#, label=f'State {j + 1}, {weight_group}')
                    legend_labels.append(f'States (23) : {weight_group}')
                    legend_handles.append(line[0])
                    #print(legend_handles)
                    legend_added.add(weight_group)
                else:
                    x_values = np.arange(i * 1001, (i + 1) * 1001)
                    y_values = states_model[j, :, i, :]  # Extract states for the current iteration
                    ax.plot(x_values, y_values, color=colors[j], alpha=0.5)
        

    # Set labels and title
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('State Values')
    ax.set_title(f'States at Each 1001 Timesteps for {29-iterations} Iterations')
    ax.legend(handles=legend_handles, labels=legend_labels)

    plt.show()
