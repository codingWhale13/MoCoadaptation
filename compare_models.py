import csv
import matplotlib.pyplot as plt
import numpy as np
import os

### CANNOT BE USED WITH OLD CSV FILES ###
### NEW VERSION ###


path='/home/oskar/Thesis/model_comparison_results' # paths need to be correct
path_link='/home/oskar/Thesis/Model_scalarized/results_with_rescaling/random seed/' # remember to set the directories correctly
newline=''

value_sums = {}
value_sums_mean = {}
link_lenghts = {}

def convert_key_to_tuple(key):
    #return a tuple of values based on which the keys are sorted
    key_values = key.split('_')
    key_values = [value for value in key_values if value] 
    #print(tuple(map(float, key_values)))
    return tuple(map(float, key_values))

for directoryname in os.listdir(path):
    directorypath = os.path.join(path, directoryname)
    #Set up dictionaries
    value_sums_mean[directoryname] = {}
    value_sums[directoryname] = {}
    if os.path.isdir(directorypath):
        for directoryname2 in os.listdir(directorypath):
            #Set new directory path
            directorypath2 = os.path.join(directorypath, directoryname2)
            #print(f"directory path 2: {directorypath2}")
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
                            value_sums_mean[directoryname][directoryname2] = {'running_speed_returns_sum_mean':np.mean(total_run_spd_reward), 'energy_consumption_returns_sum_mean':np.mean(total_energy_cons_reward)}
                            value_sums[directoryname][directoryname2] = {'running_speed_returns_sum': total_run_spd_reward , 'energy_consumption_returns_sum': total_energy_cons_reward}

#print(f" Links of the models: {link_lenghts}")

# Sort dictionaries based on keys
sorted_mean_value_sums = dict(sorted(value_sums_mean.items(), key=lambda item: convert_key_to_tuple(item[0])))
sorted_value_sums = dict(sorted(value_sums.items(), key=lambda item: convert_key_to_tuple(item[0])))

# Sort the inner dictionaries based on keys
for key, inner_dict in sorted_mean_value_sums.items():
    #print(inner_dict.items())
    sorted_mean_value_sums[key] = dict(sorted(inner_dict.items(), key=lambda item: convert_key_to_tuple(item[0])))

for key, inner_dict in sorted_value_sums.items():
    sorted_value_sums[key] = dict(sorted(inner_dict.items(), key=lambda item: convert_key_to_tuple(item[0])))

#print(f"print sorted mean value sums: {sorted_mean_value_sums }")
#print(f"print sorted value sums : {sorted_value_sums}")
#Calculate values

reward_sums = np.array([(sorted_mean_value_sums[key1][key2]['running_speed_returns_sum_mean'], 
                               sorted_mean_value_sums[key1][key2]['energy_consumption_returns_sum_mean']) for key1 in sorted_mean_value_sums.keys() for key2 in sorted_mean_value_sums[key1].keys()])

value_std = np.array([[np.std(sorted_value_sums[key1][key2]['running_speed_returns_sum'], axis=0),
                      np.std(sorted_value_sums[key1][key2]['energy_consumption_returns_sum'], axis=0)]
                     for key1 in sorted_mean_value_sums.keys() for key2 in sorted_mean_value_sums[key1].keys()])

#Plotting

fig, ax = plt.subplots()
bar_width = 0.3
off_set = 0.15
index = np.arange(len(reward_sums))

bar1 = ax.bar(index - off_set, reward_sums[:, 0], bar_width, label='Running Speed')
bar2 = ax.bar(index + off_set, reward_sums[:, 1], bar_width, label='Energy Consumption')

ax.errorbar(index - off_set, [sorted_mean_value_sums[key1][key2]['running_speed_returns_sum_mean']
                              for key1 in sorted_mean_value_sums.keys() for key2 in sorted_mean_value_sums[key1].keys()],
            yerr=[value_std[i][0] for i in range(len(value_std))], fmt='none', color='black', capsize=7)

ax.errorbar(index + off_set, [sorted_mean_value_sums[key1][key2]['energy_consumption_returns_sum_mean']
                              for key1 in sorted_mean_value_sums.keys() for key2 in sorted_mean_value_sums[key1].keys()],
            yerr=[value_std[i][1] for i in range(len(value_std))], fmt='none', color='black', capsize=7)

ax.set_xlabel('Weights')
ax.set_ylabel('Mean sums')
ax.set_title('Mean sums of Running Speed and Energy Consumption for Each Weight')
ax.set_xticks(index)
ax.set_xticklabels([key2 for key1 in sorted_mean_value_sums.keys() for key2 in sorted_mean_value_sums[key1].keys()], rotation=45, ha='right')
ax.legend()

fig2, ax2 = plt.subplots()
ax2.scatter(reward_sums[:, 0], reward_sums[:, 1], color='red', label='Reward Mean')
ax2.set_ylabel('Energy')
ax2.set_xlabel('Speed')
ax2.set_title('Mean sums of Running Speed and Energy Consumption for Each Weight')

# Annotate points on the scatter plot
for index, txt in enumerate([key2 for key1 in sorted_mean_value_sums.keys() for key2 in sorted_mean_value_sums[key1].keys()]):
    ax2.annotate(txt, (reward_sums[index, 0], reward_sums[index, 1]), textcoords="offset points", xytext=(0, 10), ha='center')

ax2.errorbar(reward_sums[:, 0], reward_sums[:, 1],
    xerr=[value_std[i][0] for i in range(len(value_std))],
    yerr=[value_std[i][1] for i in range(len(value_std))],
    fmt='none')
ax2.legend()

plt.show()