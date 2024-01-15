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


def compare_state_plot(range_iter : int, save : bool = False, save_dir = 'state_action_comparison') :
    
    """ compare states
    """ 
    
    os.makedirs(save_dir, exist_ok=True)
    
    num_states = states_model.shape[3]
    weight_categories = list(sorted_state_values.keys())
    distinct_error_colors = list(plt.get_cmap('plasma')(i / num_states) for i in range(num_states)) # viridis_r

    iter_range = range_iter

    for j in range(states_model.shape[0]): #(1):#
        figv = go.Figure()
        num_iter = np.arange(states_model.shape[3])
        for k in range(states_model.shape[3]):
                        
            color = f'rgb({distinct_error_colors[k][0]*255},{distinct_error_colors[k][1]*255},{distinct_error_colors[k][2]*255})'
            trace_name = f'State : {k}'
            
            figv.add_trace(go.Scatter(
                x=[num_iter[k]],
                y=[states_mean_array[j, k]],
                mode='markers+lines',
                name=trace_name,
                marker=dict(color=color),
                error_y=dict(
                type='data',
                array=[states_std_array[j, k]],
                visible=True,
                color=color)
                ))
            figv.update_layout(
                yaxis=dict(title='State value'),
                xaxis=dict(title='State index'),
                title=f'Comparison of weight : {weight_categories[j]} for all the states',
                hovermode='y unified',
                showlegend=True
                )
        if save:
            html_file_path = os.path.join(save_dir, f'States_{weight_categories[j]}.html')
            figv.write_html(html_file_path)
            print(f'Figure saved as {html_file_path}')
        #figv.show()
        

def compare_action_plot(range_iter : int, save : bool = False, save_dir = 'state_action_comparison') :
    
    """ compare states
    """ 
    
    os.makedirs(save_dir, exist_ok=True)
    
    num_states = actions_model.shape[3]
    weight_categories = list(sorted_state_values.keys())
    distinct_error_colors = list(plt.get_cmap('plasma')(i / num_states) for i in range(num_states)) # viridis_r

    iter_range = range_iter

    for j in range(actions_model.shape[0]): #(1):#
        figv = go.Figure()
        num_iter = np.arange(actions_model.shape[3])
        for k in range(actions_model.shape[3]):
            
            color = f'rgb({distinct_error_colors[k][0]*255},{distinct_error_colors[k][1]*255},{distinct_error_colors[k][2]*255})'
            trace_name = f'State : {k}'
                    
            figv.add_trace(go.Scatter(
                x=[num_iter[k]],
                y=[actions_mean_array[j, k]],
                mode='markers+lines',
                name=trace_name,
                marker=dict(color=color),
                error_y=dict(
                type='data',
                array=[actions_std_array[j, k]],
                visible=True,
                color=color)
                ))
            figv.update_layout(
                yaxis=dict(title='State value'),
                xaxis=dict(title='State index'),
                title=f'Comparison of weight : {weight_categories[j]} for all the states',
                hovermode='y unified',
                showlegend=True
                )
        if save:
            html_file_path = os.path.join(save_dir, f'Actions_{weight_categories[j]}.html')
            figv.write_html(html_file_path)
            print(f'Figure saved as {html_file_path}')
        figv.show()

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

    # last episode of states
    states_mean_array = np.mean(states_model[:, :, 29, :], axis=1)
    states_std_array = np.mean(states_model[:, :, 29, :], axis=1)
    
    # last episode of actions
    actions_mean_array = np.mean(actions_model[:, :, 29, :], axis=1)
    actions_std_array = np.mean(actions_model[:, :, 29, :], axis=1)

    #compare_state_plot(29, True)
    compare_action_plot(29, True)
    plt.show(block=True)