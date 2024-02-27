import csv
from colorsys import hls_to_rgb
import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.graph_objects as go


### REMEMBER TO HAVE THE BASELINE file ending as _baseline ###

#path='/home/oskar/Thesis/inter/model_comparison_results_batch_inter/vect'
#path='/home/oskar/Thesis/inter/model_comparison_results_batch_inter/steered_iterations/0.3_0.7'
path='/home/oskar/Thesis/inter/model_comparison_results_batch_inter/steered_iterations/1.0_0.0'

newline=''

#file names for link lenghts etc...
#save_file_name = 'link_lenght_batch_vect'
#OR
save_file_name = 'compare_model_iterations'


def convert_key_to_tuple(key):
    #return a tuple of values based on which the keys are sorted
    key_values = key.split(',')
    key_values = [value for value in key_values if value] 
    return tuple(map(float, key_values))


def sort_dictionaries(path):
    """ sorts through files for a given folder
    
    Returns: returns dictionaries with files values and baseline values to compare agaist
    """  
    #Set up dictionaries
    value_mean = {}
    value_std = {}
    link_lengths = {}
    
    value_mean_baseline = {}
    value_std_baseline = {}
    link_lengths_baseline = {}
    for directoryname in os.listdir(path):
        directorypath = os.path.join(path, directoryname)
        #Set up dictionaries
        if not directoryname.split("_")[-1] == 'baseline':
            directory_key = directoryname.split("__")[-1].replace(",", ", ")
            value_mean[directory_key] = {}
            value_std[directory_key] = {}
            link_lengths[directory_key] = {}
        else:
            directory_key_baseline = directoryname.split("__")[-1].replace(",", ", ").split('[')[-1:][0].split('_')[0].replace(']', "")
             # This line could lead to problems when checking results with vectorized models, since their model names dont have a space there in the naming
            value_mean_baseline[directory_key_baseline] = {}
            link_lengths_baseline[directory_key_baseline] = {}
            value_std_baseline[directory_key_baseline] = {}
        
        if os.path.isdir(directorypath):
                # Go through and save data to dictionaries
            if not directorypath.endswith("baseline"):  
                for filename in os.listdir(directorypath):
                    if filename.endswith(".csv"):
                        total_run_spd_reward = np.array([]) #reset when in new csv file
                        total_energy_cons_reward = np.array([]) 
                        link_lenghts_ind = np.array([])
                        filename_key = filename.split("_")[-1].split(".")[0] #set ind key for each csv file
                        filepath = os.path.join(directorypath, filename)
                        
                        with open(filepath, newline=newline) as file:
                            reader = csv.reader(file)
                            rows = [] # list for read values
                            # # only the last link length needs to be saved since they are the same between the test run with same model
                            for row in reader:
                                rows.append(row)
                            link_lenghts_ind = np.append(link_lenghts_ind, np.array(rows[0], dtype=float))
                            total_run_spd_reward = np.append(total_run_spd_reward, np.array(rows[1], dtype=float))
                            total_energy_cons_reward = np.append(total_energy_cons_reward, np.array(rows[2], dtype=float))
                                #values to dict
                    value_mean[directory_key][filename_key] = {'running_speed_returns_mean':np.mean(total_run_spd_reward), 'energy_consumption_returns_mean':np.mean(total_energy_cons_reward)}
                    link_lengths[directory_key][filename_key] = link_lenghts_ind
                    value_std[directory_key][filename_key] = {'running_speed_returns_std': np.std(total_run_spd_reward) , 'energy_consumption_returns_std': np.std(total_energy_cons_reward)}
            else:
                for filename in os.listdir(directorypath):
                    if filename.endswith(".csv"):
                        total_run_spd_reward = np.array([]) #reset when in new csv file
                        total_energy_cons_reward = np.array([]) 
                        link_lenghts_ind = np.array([])
                        filename_key = filename.split("_")[-1].split(".")[0] #set ind key for each csv file
                        filepath = os.path.join(directorypath, filename)
                        
                        with open(filepath, newline=newline) as file:
                            reader = csv.reader(file)
                            rows = [] # list for read values
                            # # only the last link length needs to be saved since they are the same between the test run with same model
                            for row in reader:
                                rows.append(row)
                            link_lenghts_ind = np.append(link_lenghts_ind, np.array(rows[0], dtype=float))
                            total_run_spd_reward = np.append(total_run_spd_reward, np.array(rows[1], dtype=float))
                            total_energy_cons_reward = np.append(total_energy_cons_reward, np.array(rows[2], dtype=float))
                                #values to dict
                    value_mean_baseline[directory_key_baseline][filename_key] = {'running_speed_returns_mean':np.mean(total_run_spd_reward), 'energy_consumption_returns_mean':np.mean(total_energy_cons_reward)}
                    link_lengths_baseline[directory_key_baseline][filename_key] = link_lenghts_ind
                    value_std_baseline[directory_key_baseline][filename_key] = {'running_speed_returns_std': np.std(total_run_spd_reward) , 'energy_consumption_returns_std': np.std(total_energy_cons_reward)}
                
    return value_mean, value_std, link_lengths, value_mean_baseline, value_std_baseline, link_lengths_baseline


def line_plot():
    """ Line plot for models
    """ 

    
    unique_weight_groups = sorted(set(sorted_mean.keys()), key=lambda item: convert_key_to_tuple(item.split("_")[0]))
    _, axes = plt.subplots(len(unique_weight_groups), 2, figsize=(10, 10)) # * len(unique_weight_groups)))  
    adjusted_color_dict = {weight_group: plt.get_cmap('plasma')(i / len(unique_weight_groups)) for i, weight_group in enumerate(unique_weight_groups)}
    alp_value = 0.8
    
    for i, weight in enumerate(unique_weight_groups):
        mask = [key_compare == weight for key_compare in sorted_mean.keys()]
        weight_index = np.where(mask)[0][0]
        weight_group = unique_weight_groups[np.where(mask)[0][0]]
        color = adjusted_color_dict[weight_group] 
        
        axes[i, 0].set_ylabel('Speed')
        axes[i, 0].set_xlabel('Design Cycle')
        axes[i, 0].plot(range(reward_mean.shape[1]), reward_mean[weight_index, :, 0], color=color, alpha=alp_value, label=weight_group, linewidth=2) # models per iteration to compare
        axes[i, 0].plot(range(0, 60), reward_mean_baseline[:, 0], color='darkorange', alpha=alp_value, label='baseline', linewidth=2) # baseline
        axes[i, 0].legend()
        
        axes[i, 1].set_ylabel('Energy')
        axes[i, 1].set_xlabel('Design Cycle')
        axes[i, 1].plot(range(reward_mean.shape[1]), reward_mean[weight_index, :, 1], color=color, alpha=alp_value, label=weight_group, linewidth=2) # models per iteration to compare
        axes[i, 1].plot(range(0, 60), reward_mean_baseline[:, 1], color='royalblue', alpha=alp_value, label='baseline', linewidth=2) # baseline
        axes[i, 1].legend()

    
def link_length_plot(save_file : bool = False, save_dir = 'link_length_comparison_results'):
    """ link length plot
    Set to 'True' to save plots into html files or 'False' to not save
    """  
    weight_categories = list(sorted_link_lengths.keys())
    distinct_error_colors = list(plt.get_cmap('plasma')(i / len(weight_categories)) for i, _ in enumerate(weight_categories)) # viridis_r

    os.makedirs(save_dir, exist_ok=True)

    for j in range(link_lengths_array.shape[2]):
        figv = go.Figure()
        figm = go.Figure()
        index_link_length = np.arange(link_lengths_array.shape[0])
        index_weigths = np.arange(link_lengths_array.shape[0])
        for i, weight_category in enumerate(weight_categories):
            
            color = f'rgb({distinct_error_colors[i][0]*255},{distinct_error_colors[i][1]*255},{distinct_error_colors[i][2]*255})'
            
            # MEAN AND STANDARD DEVIATION
            figm.add_trace(go.Scatter(
                    x=[weight_categories[i]],#weight_categories, #+ group_offset,
                    y=[link_lengths_mean_array[i, j]],
                    mode='markers+lines',
                    name=f'Weight : {weight_category} ',
                    marker=dict(color=color),
                    error_y=dict(
                    type='data',
                    array=[ci_link_length[i, j]],
                    visible=True,
                    color=color)
                ))
            figm.update_layout(
                        yaxis=dict(title='Link lenght',tickvals=index_link_length),
                        xaxis=dict(title='weights', tickvals=index_weigths, ticktext=weight_categories),
                        title=f'Comparison of Mean Link Lengths for Link {j + 1}',
                        showlegend=True
            )
            
            # REGULAR VALUES
            for _, y_value in enumerate(link_lengths_array[i, :, j]):
                figv.add_trace(go.Scatter(
                    x=[weight_categories[i]],  # Use a list with a single value
                    y=[y_value],
                    mode='markers+lines',
                    name=f'Weight : {weight_category}',
                    marker=dict(color=color),
                    line=dict(color=color)
                ))
                
            figv.update_layout(
                        yaxis=dict(title='Link lenght',tickvals=index_link_length),
                        xaxis=dict(title='weights', tickvals=index_weigths, ticktext=weight_categories),
                        title=f'Comparison of Link Lengths for Link {j + 1}',
                        showlegend=True
            )
        if save_file: 
            #save as html for comparisons
            html_file_path_mean = os.path.join(save_dir, f'Link_length_{j + 1}_mean_values.html')
            figm.write_html(html_file_path_mean)
            print(f'Figure saved as {html_file_path_mean}')
            html_file_path_reg = os.path.join(save_dir, f'Link_length_{j + 1}_values.html')
            figv.write_html(html_file_path_reg)
            print(f'Figure saved as {html_file_path_reg}')
            #save the means as pdf
            pdf_file_path_mean = os.path.join(save_dir, f'Link_length_{j + 1}_mean_values_{save_file_name.split("_")[-1]}_.pdf')
            figm.write_image(pdf_file_path_mean, format='pdf')
            print(f'Figure saved as {pdf_file_path_mean}')
            #save the values as pdf
            pdf_file_path_values = os.path.join(save_dir, f'Link_length_{j + 1}_values_{save_file_name.split("_")[-1]}_.pdf')
            figv.write_image(pdf_file_path_values, format='pdf')
            print(f'Figure saved as {pdf_file_path_values}')
        #figm.show() # uncomment to see the plots 
        #figv.show() # uncomment to see the plots 

if __name__ == "__main__":
    
    sample_count = 5 # ADD how many samples you have (test runs per model weight and seed) or (test runs per loaded model and weight (at least for now the seed isnt considered))
    value_mean, value_std, link_lengths, value_mean_baseline, value_std_baseline, link_lengths_baseline = sort_dictionaries(path) # If you're analysing models trained with pre-trained model, pass 'True' else 'False'
    
    # Sort dictionaries based on keys
    sorted_mean = dict(sorted(value_mean.items(), key=lambda item: convert_key_to_tuple(item[0].split("_")[0])))
    sorted_std = dict(sorted(value_std.items(), key=lambda item: convert_key_to_tuple(item[0].split("_")[0])))
    sorted_link_lengths = dict(sorted(link_lengths.items(), key=lambda item: convert_key_to_tuple(item[0].split("_")[0]))) #Sort link lengths
    
    sorted_mean_baseline = dict(sorted(value_mean_baseline.items(), key=lambda item: convert_key_to_tuple(item[0].split("_")[0])))
    sorted_std_baseline = dict(sorted(value_std_baseline.items(), key=lambda item: convert_key_to_tuple(item[0].split("_")[0])))
    sorted_link_lengths_baseline = dict(sorted(link_lengths_baseline.items(), key=lambda item: convert_key_to_tuple(item[0].split("_")[0]))) #Sort link lengths

    # Sort the inner dictionaries based on keys - baseline
    # mean values
    for key, inner_dict in sorted_mean_baseline.items():
        #print(inner_dict.items())
        sorted_mean_baseline[key] = dict(sorted(inner_dict.items(), key=lambda item: convert_key_to_tuple(item[0])))
    # sums
    for key, inner_dict in sorted_std_baseline.items():
        sorted_std_baseline[key] = dict(sorted(inner_dict.items(), key=lambda item: convert_key_to_tuple(item[0]))) 
    # link lengths
    for key, inner_dict in sorted_link_lengths_baseline.items():
        sorted_link_lengths_baseline[key] = dict(sorted(inner_dict.items(), key=lambda item: convert_key_to_tuple(item[0])))

    # Sort the inner dictionaries based on keys
    # mean values
    for key, inner_dict in sorted_mean.items():
        #print(inner_dict.items())
        sorted_mean[key] = dict(sorted(inner_dict.items(), key=lambda item: convert_key_to_tuple(item[0])))
    # sums
    for key, inner_dict in sorted_std.items():
        sorted_std[key] = dict(sorted(inner_dict.items(), key=lambda item: convert_key_to_tuple(item[0]))) 
    # link lengths
    for key, inner_dict in sorted_link_lengths.items():
        sorted_link_lengths[key] = dict(sorted(inner_dict.items(), key=lambda item: convert_key_to_tuple(item[0])))

    #Calculate values to arrays
    #baseline
    reward_mean_baseline = np.array([(sorted_mean_baseline[key1][key2]['running_speed_returns_mean'], 
                                sorted_mean_baseline[key1][key2]['energy_consumption_returns_mean']) for key1 in sorted_mean_baseline.keys() for key2 in sorted_mean_baseline[key1].keys()])

    reward_std_baseline = np.array([(sorted_std_baseline[key1][key2]['running_speed_returns_std'], 
                                sorted_std_baseline[key1][key2]['energy_consumption_returns_std']) for key1 in sorted_std_baseline.keys() for key2 in sorted_std_baseline[key1].keys()])
    
    # values to compare
    reward_mean = np.array([[(sorted_mean[key1][key2]['running_speed_returns_mean'], 
                                sorted_mean[key1][key2]['energy_consumption_returns_mean']) for key2 in sorted_mean[key1].keys()] for key1 in sorted_mean.keys()])

    reward_std = np.array([[(sorted_std[key1][key2]['running_speed_returns_std'], 
                                sorted_std[key1][key2]['energy_consumption_returns_std']) for key2 in sorted_std[key1].keys()] for key1 in sorted_mean.keys()])
    
    #baseline
    link_lengths_array = np.array([[weights[key] for key in weights] for weights in sorted_link_lengths_baseline.values()])
    #link_lengths_mean_array = np.array([np.mean(list(weights.values()), axis=0) for weights in sorted_link_lengths_baseline.values()])
    #link_lengths_std_array = np.array([np.std(list(weights.values()), axis=0) for weights in sorted_link_lengths_baseline.values()])

    # model values
    link_lengths_array = np.array([[weights[key] for key in weights] for weights in sorted_link_lengths.values()])
    link_lengths_mean_array = np.array([np.mean(list(weights.values()), axis=0) for weights in sorted_link_lengths.values()])
    link_lengths_std_array = np.array([np.std(list(weights.values()), axis=0) for weights in sorted_link_lengths.values()])
    
    #Set condidence inverval
    confidency_interval = 1.96 #3.89 #1.96#2.5576 #1.96  # Give confidence interval
    #ci_running_speed = confidency_interval * (np.array([reward_std[i][0] for i in range(len(reward_std))]) / np.sqrt(sample_count))  # sample size is 5 since we have 5 different test runs  #np.sqrt(len(reward_std)))
    #ci_energy_consumption = confidency_interval * (np.array([reward_std[i][1] for i in range(len(reward_std))]) / np.sqrt(sample_count))
    
    sem_link_length = link_lengths_std_array / np.sqrt(56) #5 for vect or 7 for steered models per weight # sample size (amount of trained models per weight and seed) -> regular model or (trained models per loaded weight in weight) -> loaded model
    ci_link_length = confidency_interval * sem_link_length
    
    print() 
    
    line_plot()
    # if save_file_name:
    #     link_length_plot(True, save_file_name)
    # else:
    #     link_length_plot(False)
    plt.show(block=True)