import csv
import matplotlib.pyplot as plt
import numpy as np
import os

### CANNOT BE USED WITH OLD CSV FILES ###
### NEW VERSION ###

def convert_key_to_tuple(key):
    #return a tuple of values based on which the keys are sorted
    key_values = key.split('_')
    key_values = [value for value in key_values if value] 
    #print(tuple(map(float, key_values)))
    return tuple(map(float, key_values))

path='/home/oskar/Thesis/model_comparison_results' # paths need to be correct
path_link='/home/oskar/Thesis/Model_scalarized/results_with_rescaling/random seed/' # remember to set the directories correctly
newline=''

value_sums = {}
value_sums_mean = {}
link_lengths = {}

#if __name__ == "__main__": 

for directoryname in os.listdir(path):
    directorypath = os.path.join(path, directoryname)
    #Set up dictionaries
    value_sums_mean[directoryname] = {}
    value_sums[directoryname] = {}
    link_lengths[directoryname] = {}
    if os.path.isdir(directorypath):
        for directoryname2 in os.listdir(directorypath):
            #Set new directory path
            directorypath2 = os.path.join(directorypath, directoryname2)
            
            directory_keyname = directoryname2.split('[')[-1:]
            directory_keyname = directory_keyname[0].replace("]", "_").replace(", ", "_")
            #print(f"directory path 2: {directorypath2}")
            # Go through and save data to dictionaries
            if os.path.isdir(directorypath2):
                total_run_spd_reward = np.array([]) #reset when in new file
                total_energy_cons_reward = np.array([])
                link_lenghts_ind = np.array([])
                
                for filename in os.listdir(directorypath2):
                    if filename.endswith(".csv"):
                        #print(filename)
                        filepath = os.path.join(directorypath2, filename)
                        with open(filepath, newline=newline) as file:
                            reader = csv.reader(file)
                            rows = [] # list for read values
                            running_speed_reward = np.array([])#reset when in new file
                            energy_consumption_reward = np.array([])
                            
                            total_run_spd_reward = np.array([]) #reset when in new directory
                            total_energy_cons_reward = np.array([])
                            link_lenghts_ind = np.array([])
                            for row in reader:
                                rows.append(row)
                                
                            run_speed_sum_reward_sum = np.sum(np.array(rows[1], dtype=float))
                            energy_cons_reward_sum = np.sum(np.array(rows[2], dtype=float))
                            link_lenghts_ind = np.append(link_lenghts_ind, np.array(rows[0], dtype=float))
                            total_run_spd_reward = np.append(total_run_spd_reward, run_speed_sum_reward_sum)
                            total_energy_cons_reward = np.append(total_energy_cons_reward, energy_cons_reward_sum)
                            value_sums_mean[directoryname][directory_keyname] = {'running_speed_returns_sum_mean':np.mean(total_run_spd_reward), 'energy_consumption_returns_sum_mean':np.mean(total_energy_cons_reward)}
                            link_lengths[directoryname][directory_keyname] = link_lenghts_ind
                            value_sums[directoryname][directory_keyname] = {'running_speed_returns_sum': total_run_spd_reward , 'energy_consumption_returns_sum': total_energy_cons_reward}

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

# link_lengths_array = np.array([list(sorted_link_lengths.values())])
# print(link_lengths_array)
# print(f" Links of the models: {sorted_link_lengths}")

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
#ax2.scatter(reward_sums[:, 0], reward_sums[:, 1], color='orange', label='Reward Mean') # LEGACY not need
ax2.set_ylabel('Energy')
ax2.set_xlabel('Speed')
ax2.set_title('Mean sums of Running Speed and Energy Consumption for Each Weight')


unique_weight_groups = sorted(set([key1 for key1 in sorted_mean_value_sums.keys()]))
print(unique_weight_groups)

#color_dict = {weight_group: plt.cm.viridis(i / len(unique_weight_groups)) for i, weight_group in enumerate(unique_weight_groups)}
color_dict = {weight_group: plt.get_cmap('rainbow')(i / len(unique_weight_groups)) for i, weight_group in enumerate(unique_weight_groups)}
#"color_dict = {weight_group: plt.get_cmap('tab10')(i) for i, weight_group in enumerate(unique_weight_groups)} # WORK WITH ONLY 10 colors # use viridian if 


legend_added = {} # keep track of added legends for weight groups
print(color_dict)
#colors = plt.cm.viridis(np.linspace(0, 1, len(unique_weight_groups)))
#print(colors)

for index, (key1, key2) in enumerate([(key1, key2) for key1 in sorted_mean_value_sums.keys() for key2 in sorted_mean_value_sums[key1].keys()]):
    #print(index)
    #print((key1, key2))
    #weight_group = key1  # Assuming the first part of the key represents the weight group
    mask = [key_compare == key1 for key_compare in sorted_mean_value_sums.keys()]
    #print(mask)
    weight_group = unique_weight_groups[np.where(mask)[0][0]]  # Find the index where the mask is True
    #ax2.scatter(reward_sums[index, 0], reward_sums[index, 1], color=color_dict[weight_group], label=weight_group)
    #ax2.scatter(reward_sums[index, 0], reward_sums[index, 1], color=color_dict[unique_weight_groups[0]], label=key1)
    #ax2.scatter(reward_sums[index, 0], reward_sums[index, 1], color=colors[weight_group], label=weight_group)
    if weight_group not in legend_added:
        ax2.scatter(reward_sums[index, 0], reward_sums[index, 1], color=color_dict[weight_group], label=weight_group)
        legend_added[weight_group] = True
    else:
        ax2.scatter(reward_sums[index, 0], reward_sums[index, 1], color=color_dict[weight_group])


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

plt.show()
