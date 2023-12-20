import csv
import matplotlib.pyplot as plt
import numpy as np
import os

##### This is a dated ##### should work through

def convert_key_to_tuple(key):
    #return a tuple of values based on which the keys are sorted
    key_values = key.split('_')
    key_values = [value for value in key_values if value] 
    #print(tuple(map(float, key_values)))
    return tuple(map(float, key_values))



path='/home/oskar/Thesis/episodic_rewards_scaled'
#path='/home/oskar/Thesis/episodic_rewards_unscaled'
newline=''

value_sums = {}
value_sums_mean = {}
link_lenghts = {}


for directoryname in os.listdir(path):
    #print(f"dir name found: {directoryname}")
    directorypath = os.path.join(path, directoryname)
    if os.path.isdir(directorypath):
        # total_run_spd_reward = np.array([]) #reset when in new directory
        # total_energy_cons_reward = np.array([])
        # link_lenghts_ind = np.array([])
        for filename in os.listdir(directorypath):
            if filename.endswith(".csv"):
                filepath = os.path.join(directorypath, filename)
                with open(filepath, newline=newline) as file:
                    reader = csv.reader(file)
                    rows = []
                    total_run_spd_reward = np.array([]) #reset when in new file
                    total_energy_cons_reward = np.array([])
                    link_lenghts_ind = np.array([])
                    running_speed_reward = np.array([])#reset when in new file
                    energy_consumption_reward = np.array([])
                    for row in reader:
                        rows.append(row)
                    run_speed_sum_reward_sum = np.sum(np.array(rows[1], dtype=float))
                    energy_cons_reward_sum = np.sum(np.array(rows[2], dtype=float))
                    link_lenghts_ind = np.append(link_lenghts_ind, np.array(rows[0], dtype=float))
                    total_run_spd_reward = np.append(total_run_spd_reward, run_speed_sum_reward_sum)
                    total_energy_cons_reward = np.append(total_energy_cons_reward, energy_cons_reward_sum)
                link_lenghts[directoryname] = {'link_lenghts' : link_lenghts_ind}
                value_sums_mean[directoryname] = {'running_speed_returns_sum_mean':np.mean(total_run_spd_reward), 'energy_consumption_returns_sum_mean':np.mean(total_energy_cons_reward)}
                value_sums[directoryname] = {'running_speed_returns_sum': total_run_spd_reward , 'energy_consumption_returns_sum': total_energy_cons_reward}
           
value_sums_mean_sorted = dict(sorted(value_sums_mean.items())) # sort keys  
#print(value_sums_mean_sorted)
#directories = list(sorted(value_sums_mean.keys())) # ORIG
#print(directories)
#scaled
print(f"link lenghts: {link_lenghts}")

#key_order = ['0.0_1.0', '0.01_0.99'] + [key for key in value_sums_mean_sorted if key not in ['0.0_1.0', '0.01_0.99', '1.0_0.0', '0.99_0.01']] + ['0.99_0.01', '1.0_0.0'] # switch places or 0.0_1.0 and 0.01_0.99
#unscaled
#key_order = ['0.0_1.0'] + [key for key in value_sums_mean_sorted if key not in ['0.0_1.0', '1.0_0.0']] + ['1.0_0.0'] # switch places or 0.0_1.0 and 0.01_0.99
#print(key_order)
key_order = [key for key in value_sums_mean_sorted]


value_sums_mean_sorted = {key: value_sums_mean_sorted[key] for key in key_order}

value_std = {index: [np.std(value_sums[index]['running_speed_returns_sum'], axis=0),
                    np.std(value_sums[index]['energy_consumption_returns_sum'], axis=0)]
             for index in key_order}

running_speed_sums = [item['running_speed_returns_sum_mean'] for item in value_sums_mean_sorted.values()] # ORIG
energy_cons_sums = [item['energy_consumption_returns_sum_mean'] for item in value_sums_mean_sorted.values()]

#Plotting

fig, ax = plt.subplots()
bar_width = 0.3
off_set = 0.15
index = np.arange(len(key_order))

bar1 = ax.bar(index - off_set , running_speed_sums, bar_width, label='Running Speed')
bar2 = ax.bar(index + off_set , energy_cons_sums, bar_width, label='Energy Consumption')
#DOES not work if the energy has smaller values than the running speed!
#bar2 = ax.bar(index, [energy_cons_sums[value]-running_speed_sums[value] for value, _ in enumerate(energy_cons_sums)], bar_width, label='Energy Consumption', bottom=running_speed_sums) # less efficient with enumerate
#bar2 = ax.bar(index, [energy - running_speed for energy, running_speed in zip(energy_cons_sums, running_speed_sums)], bar_width, label='Energy Consumption', bottom=running_speed_sums)

#std error bars 
ax.errorbar(index - off_set , [value_sums_mean[index]['running_speed_returns_sum_mean'] for index in key_order],
            yerr=[value_std[index][0] for index in key_order], fmt='none', color='black', capsize=7)
ax.errorbar(index + off_set , [value_sums_mean[index]['energy_consumption_returns_sum_mean'] for index in key_order],
            yerr=[value_std[index][1] for index in key_order], fmt='none', color='black', capsize=7)

ax.set_xlabel('Weights')
ax.set_ylabel('Mean sums')
ax.set_title('Mean sums of Running Speed and Energy Consumption for Each Weight')
ax.set_xticks(index)
ax.set_xticklabels(key_order)
ax.legend()

fig2, ax2 = plt.subplots()
ax2.scatter(running_speed_sums, energy_cons_sums, color='red')
ax2.set_ylabel('Energy')
ax2.set_xlabel('Speed')
ax2.set_title('Mean sums of Running Speed and Energy Consumption for Each Weight')

for index, weight in enumerate(key_order):
    ax2.annotate(key_order[index], (running_speed_sums[index],energy_cons_sums[index]), textcoords="offset points", xytext=(0,10), ha='center')
#print(key_order)
# ax2.errorbar(running_speed_sums, energy_cons_sums,
#             xerr=[value_std[index][0] for index in key_order],
#             yerr=[value_std[index][1] for index in key_order],
#             fmt='none', color='black', marker='x',label="Bar plot") # ORIG
ax2.errorbar(running_speed_sums, energy_cons_sums,
            xerr=[value_std[index][0] for index in key_order],
            yerr=[value_std[index][1] for index in key_order],
            fmt=':b',label="Bar plot")
ax2.legend()

plt.show()