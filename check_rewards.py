import csv
import matplotlib.pyplot as plt
import numpy as np
import os

path='/home/oskar/Thesis/episodic_rewards_scaled'
#'/home/oskar/Work/Changes/MoCoadaptation/data_exp_sac_pso_sim/Wed_Nov_15_11:56:27_2023__aa88d15c'
#'/home/oskar/Thesis/episodic_rewards/'
newline=''

value_sums = {}
value_sums_mean = {}

for directoryname in os.listdir(path):
    #print(f"dir name found: {directoryname}")
    directorypath = os.path.join(path, directoryname)
    if os.path.isdir(directorypath):
        total_run_spd_reward = np.array([]) #reset when in new directory
        total_energy_cons_reward = np.array([])
        for filename in os.listdir(directorypath):
            if filename.endswith(".csv"):
                filepath = os.path.join(directorypath, filename)
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
                    # print("\n REWARDS:")
                    # print(total_run_spd_reward)
                    # print(total_energy_cons_reward)
                #print()
                #print(total_run_spd_reward, total_energy_cons_reward)
                #print()
                value_sums_mean[directoryname] = {'running_speed_returns_sum_mean':np.mean(total_run_spd_reward), 'energy_consumption_returns_sum_mean':np.mean(total_energy_cons_reward)}
                value_sums[directoryname] = {'running_speed_returns_sum': total_run_spd_reward , 'energy_consumption_returns_sum': total_energy_cons_reward}
                #print(f"running_speed_returns_sum_mean': {np.mean(total_run_spd_reward)}, 'energy_consumption_returns_sum_mean': {np.mean(total_energy_cons_reward)}")
                
value_sums_mean_sorted = dict(sorted(value_sums_mean.items()))
directories = sorted(list(value_sums_mean.keys()))

value_std = {index: [np.std(value_sums[index]['running_speed_returns_sum'], axis=0),
                    np.std(value_sums[index]['energy_consumption_returns_sum'], axis=0)]
             for index in sorted(list(value_sums.keys()))}

running_speed_sums = [item['running_speed_returns_sum_mean'] for item in value_sums_mean_sorted.values()]
energy_cons_sums = [item['energy_consumption_returns_sum_mean'] for item in value_sums_mean_sorted.values()]

#Plotting
fig, ax = plt.subplots()
bar_width = 0.3
index = np.arange(len(directories))

bar1 = ax.bar(index - 0.15, running_speed_sums, bar_width, align='center', label='Running Speed', alpha=1)
bar2 = ax.bar(index + 0.15, energy_cons_sums, bar_width, align='center', label='Energy Consumption', alpha=1)
#DO not work if the energy has smaller values than the running speed!
#bar2 = ax.bar(index, [energy_cons_sums[value]-running_speed_sums[value] for value, _ in enumerate(energy_cons_sums)], bar_width, label='Energy Consumption', bottom=running_speed_sums) # less efficient with enumerate
#bar2 = ax.bar(index, [energy - running_speed for energy, running_speed in zip(energy_cons_sums, running_speed_sums)], bar_width, label='Energy Consumption', bottom=running_speed_sums)

#std error bars 
ax.errorbar(index - bar_width / 2, [value_sums_mean[index]['running_speed_returns_sum_mean'] for index in directories],
            yerr=[value_std[index][0] for index in directories], fmt='none', color='black', capsize=5)
ax.errorbar(index + bar_width / 2, [value_sums_mean[index]['energy_consumption_returns_sum_mean'] for index in directories],
            yerr=[value_std[index][1] for index in directories], fmt='none', color='black', capsize=5)

ax.set_xlabel('Weights')
ax.set_ylabel('Mean sums')
ax.set_title('Mean sums of Running Speed and Energy Consumption for Each Weight')
ax.set_xticks(index)
ax.set_xticklabels(directories)
ax.legend()

plt.show()