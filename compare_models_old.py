import csv
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import os
import matplotlib.cm as cm
import matplotlib.patches as mpatches

### Version for the csv files which dont have morphology saved on in the rewards csv files ###
### OLD ###


path='/home/oskar/Thesis/model_comparison_results_old' # paths need to be correct
path_link='/home/oskar/Thesis/Model_scalarized/results_with_rescaling/random_seed/' # remember to set the directories correctly
newline=''

value_sums = {}
value_sums_mean = {}
link_lengths = {}

def convert_key_to_tuple(key):
    #return a tuple of values based on which the keys are sorted
    key_values = key.split('_')
    key_values = [value for value in key_values if value] 
    #print(tuple(map(float, key_values)))
    return tuple(map(float, key_values))

def check_path(path_link, weightdir):
    """ Check the path to model

    Args:
        path_link (_type_): _description_
        weightdir (_type_): _description_

    Returns: path for correct weight
    """    
    #return the correct path with correct directory per weight
    path_dir = os.path.join(path_link, weightdir)
    return path_dir

def find_checkpoint(path_to_directory):
    """ Find the checkpoint for the model

    Returns: returns the int value of the last checkpoint or None
    """    
    checkpoints = []
    for file in os.listdir(path_to_directory):    
        if file.endswith('.csv'):
            checkpoint = int(file.split('_')[-1][:-4])
            checkpoints.append(checkpoint)
    if checkpoints:
        return max(checkpoints)
    else:
        return None

def read_morphology(morphologydirname, weightdir, morphology_number) -> list:
    
    """ Returns a list of values read from cvs file per row
    
    Returns:
        a list containing csv file values
    """    
    rows = []
    path_dir = check_path(path_link, weightdir)
    morphology_number = str(morphology_number)  + ".csv"
    for dirname in os.listdir(path_dir):
        if dirname == morphologydirname:
            filepath = os.path.join(path_dir, dirname)
            for filename in os.listdir(filepath):
                #print(morphology_number)
                if filename.endswith(morphology_number):
                    filename2 = os.path.join(filepath, filename)
                    #print(f" Selected morphology number : {filename}")
                    with open(filename2, newline=newline) as file:
                        reader = csv.reader(file)
                        for row in reader:
                            rows.append(row)
    return rows


for directoryname in os.listdir(path):
    directorypath = os.path.join(path, directoryname)
    #Set up dictionaries
    #dirkey = 
    value_sums_mean[directoryname] = {}
    value_sums[directoryname] = {}
    if os.path.isdir(directorypath):
        for directoryname2 in os.listdir(directorypath):
            #Set new directory path
            directorypath2 = os.path.join(directorypath, directoryname2)
            #print(f"directory path 2: {directorypath2}")
            #cut directoryname2 to fit better as key name
            directory_keyname = directoryname2.split('[')[-1:]
            directory_keyname = directory_keyname[0].replace("]", "_").replace(", ", "_")
            
            morphologydir = directoryname2[:-2] #you need this drop last characters in name two for some files
            # since link lenghts are saved in different files, we will need to know the weights to access the correct directory
            weightdir = morphologydir.split('[')[-1]
            weightdir = weightdir[:-1]
            weightdir = weightdir.replace(", ", "_")
            
            #find correct checkpoint for morphology
            checkpoint_path = os.path.join(path_link, weightdir)
            morphology_num = find_checkpoint(os.path.join(checkpoint_path, morphologydir))
            model_file = read_morphology(morphologydir, weightdir, morphology_num) # read model csv file as list
            #print(model_file)
            link_lengths_ind = np.array(model_file[1], dtype=float) # index link lengths from the file
            link_lengths[directory_keyname] = link_lengths_ind 
            
            # Go through and save data to dictionaries
            if os.path.isdir(directorypath2):
                total_run_spd_reward = np.array([]) #reset when in new directory
                total_energy_cons_reward = np.array([])
                for filename in os.listdir(directorypath2):
                    if filename.endswith(".csv"):
                        #print(filename)
                        filepath = os.path.join(directorypath2, filename)
                        with open(filepath, newline=newline) as file:
                            reader = csv.reader(file)
                            running_speed_reward = np.array([])#reset when in new file
                            energy_consumption_reward = np.array([])
                            for row in reader:
                                running_speed_reward = np.append(running_speed_reward, float(row[0]))
                                energy_consumption_reward = np.append(energy_consumption_reward, float(row[1]))
                            run_speed_sum_reward_sum = np.sum(running_speed_reward)
                            energy_cons_reward_sum = np.sum(energy_consumption_reward)
                            total_run_spd_reward = np.append(total_run_spd_reward, run_speed_sum_reward_sum)
                            total_energy_cons_reward = np.append(total_energy_cons_reward, energy_cons_reward_sum)
                            value_sums_mean[directoryname][directory_keyname] = {'running_speed_returns_sum_mean':np.mean(total_run_spd_reward), 'energy_consumption_returns_sum_mean':np.mean(total_energy_cons_reward)}
                            value_sums[directoryname][directory_keyname] = {'running_speed_returns_sum': total_run_spd_reward , 'energy_consumption_returns_sum': total_energy_cons_reward}


# Sort dictionaries based on keys
sorted_mean_value_sums = dict(sorted(value_sums_mean.items(), key=lambda item: convert_key_to_tuple(item[0])))
sorted_value_sums = dict(sorted(value_sums.items(), key=lambda item: convert_key_to_tuple(item[0])))

#Sort link lenghts
sorted_link_lengths = dict(sorted(link_lengths.items(), key=lambda item: convert_key_to_tuple(item[0])))
labels_links = list(sorted_link_lengths.keys())

link_lengths_array = np.array([list(sorted_link_lengths.values())])

print(link_lengths_array)
#link_lengths_array_mean = np.mean(sorted_link_lengths.values())
#print(link_lengths_array_mean)


# Sort the inner dictionaries based on keys
for key, inner_dict in sorted_mean_value_sums.items():
    #print(inner_dict.items())
    sorted_mean_value_sums[key] = dict(sorted(inner_dict.items(), key=lambda item: convert_key_to_tuple(item[0])))
for key, inner_dict in sorted_value_sums.items():
    sorted_value_sums[key] = dict(sorted(inner_dict.items(), key=lambda item: convert_key_to_tuple(item[0]))) 

print(f" Links of the models: {sorted_link_lengths}")

#Calculate values

reward_sums = np.array([(sorted_mean_value_sums[key1][key2]['running_speed_returns_sum_mean'], 
                               sorted_mean_value_sums[key1][key2]['energy_consumption_returns_sum_mean']) for key1 in sorted_mean_value_sums.keys() for key2 in sorted_mean_value_sums[key1].keys()])

value_std = np.array([[np.std(sorted_value_sums[key1][key2]['running_speed_returns_sum'], axis=0),
                      np.std(sorted_value_sums[key1][key2]['energy_consumption_returns_sum'], axis=0)]
                     for key1 in sorted_mean_value_sums.keys() for key2 in sorted_mean_value_sums[key1].keys()])

#bar plot
fig, ax = plt.subplots()
bar_width = 0.3
off_set = 0.15
index = np.arange(len(reward_sums))

bar1 = ax.bar(index - off_set, reward_sums[:, 0], bar_width, label='Running Speed')
bar2 = ax.bar(index + off_set, reward_sums[:, 1], bar_width, label='Energy Consumption')

ax.errorbar(index - off_set, [sorted_mean_value_sums[key1][key2]['running_speed_returns_sum_mean']
                              for key1 in sorted_mean_value_sums.keys() for key2 in sorted_mean_value_sums[key1].keys()],
            yerr=[value_std[i][0] for i in range(len(value_std))], fmt='none', color='black', capsize=5)

ax.errorbar(index + off_set, [sorted_mean_value_sums[key1][key2]['energy_consumption_returns_sum_mean']
                              for key1 in sorted_mean_value_sums.keys() for key2 in sorted_mean_value_sums[key1].keys()],
            yerr=[value_std[i][1] for i in range(len(value_std))], fmt='none', color='black', capsize=5)

#scatter plot
ax.set_xlabel('Weights')
ax.set_ylabel('Mean sums')
ax.set_title('Mean sums of Running Speed and Energy Consumption for Each Weight')
ax.set_xticks(index)
ax.set_xticklabels([key2 for key1 in sorted_mean_value_sums.keys() for key2 in sorted_mean_value_sums[key1].keys()], rotation=45, ha='right')
ax.legend()

fig2, ax2 = plt.subplots()
ax2.scatter(reward_sums[:, 0], reward_sums[:, 1], color='orange', label='Reward Mean')
ax2.set_ylabel('Energy')
ax2.set_xlabel('Speed')
ax2.set_title('Mean sums of Running Speed and Energy Consumption for Each Weight')

# Annotate points on the scatter plot
for index, txt in enumerate([key2 for key1 in sorted_mean_value_sums.keys() for key2 in sorted_mean_value_sums[key1].keys()]):
    ax2.annotate(txt, (reward_sums[index, 0], reward_sums[index, 1]), textcoords="offset points", xytext=(0, 10), ha='center')

ax2.errorbar(reward_sums[:, 0], reward_sums[:, 1],
    xerr=[value_std[i][0] for i in range(len(value_std))],
    yerr=[value_std[i][1] for i in range(len(value_std))],
    fmt=':b')
ax2.legend()

#DOES NOT WORK PROPERLY

# test_amount = 5
# num_figures = len(labels_links) // test_amount

# link_lengths_array_mean = np.array([])
# for i in range(num_figures) : link_lengths_array_mean = np.append(link_lengths_array_mean,np.mean(link_lengths_array[0][0+test_amount*i:test_amount+test_amount*i], axis=0))

# # Plotting link lengths and adding average link lengths as a red line
# for fig_num in range(num_figures):
#     plt.figure(figsize=(15, 12))  # Adjust the figure size as needed
#     for i in range(test_amount):
#         index = fig_num * test_amount + i
#         if index < len(labels_links):
#             plt.subplot(test_amount, 1, i + 1)
#             #plt.bar(range(len(link_lengths_array[index])), link_lengths_array[i], label=labels_links[index])
#             plt.axhline(y=link_lengths_array_mean[index], color='red', linestyle='--', label='Average')
#             plt.xlabel('Link Index')
#             plt.ylabel('Link Lengths')
#             plt.title(f'Link Lengths for {labels_links[index]}', color='orange')
#             plt.legend()

# plt.show()

    