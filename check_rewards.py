import csv
import matplotlib.pyplot as plt
import numpy as np
import os


path='/home/oskar/Thesis/episodic_rewards_scaled'
#'/home/oskar/Work/Changes/MoCoadaptation/data_exp_sac_pso_sim/Wed_Nov_15_11:56:27_2023__aa88d15c'
#'/home/oskar/Thesis/episodic_rewards/'
newline=''

value_sums = {}
list_of_names = []

for directoryname in os.listdir(path):
    list_of_names.append(directoryname)
    print(f"dir name found: {directoryname}")
    directorypath = os.path.join(path, directoryname)
    if os.path.isdir(directorypath):
        for filename in os.listdir(directorypath):
            if filename.endswith(".csv"):
                filepath = os.path.join(directorypath, filename)
                running_speed_reward = np.array([])
                energy_consumption_reward = np.array([])
                total_run_spd_reward = np.array([])
                total_energy_cons_reward = np.array([])
                
                with open(filepath, newline=newline) as file:
                    reader = csv.reader(file)
                    for row in reader:
                        running_speed_reward = np.append(running_speed_reward, float(row[0]))
                        energy_consumption_reward = np.append(energy_consumption_reward, float(row[1]))
                        
                    run_speed_sum_reward_sum = np.sum(running_speed_reward)
                    energy_cons_reward_sum = np.sum(energy_consumption_reward)
                    total_run_spd_reward = np.append(total_run_spd_reward, run_speed_sum_reward_sum)
                    total_energy_cons_reward= np.append(total_energy_cons_reward, energy_cons_reward_sum)
                value_sums[directoryname] = {'running_speed_returns_sum_mean':np.mean(total_run_spd_reward), 'energy_consumption_returns_sum_mean':np.mean(total_energy_cons_reward)}

print(list_of_names)
print()
# #print("Mean rewards:")  
# print("Sum of rewards")
# print(total_run_spd_reward)  
# print(total_energy_cons_reward)         
# #print(total_run_spd_reward)
# #print(total_energy_cons_reward)
# print(f"Mean of the runs of running speed rewards: {np.mean(total_run_spd_reward)}")
# print(f"Mean of the runs of energy consumption rewards: {np.mean(total_energy_cons_reward)}")
# print(f"Ratio of means: {np.mean(total_run_spd_reward)/np.mean(total_energy_cons_reward)}")

# Plotting
# plt.plot(value_sums[list_of_names]['running_speed_returns_sum_mean'], label='Running Speed Reward')
# plt.plot(value_sums[list_of_names]['energy_consumption_returns_sum_mean'], label='Energy Consumption Reward')
# plt.xlabel('Run Index')
# plt.ylabel('Mean Reward')
# plt.title('Mean Rewards Across Runs')
# plt.legend()
# plt.show()
# print()
  
  # Extract directory names and corresponding sums
directories = list(value_sums.keys())
print(directories)
running_speed_sums = [item['running_speed_returns_sum_mean'] for item in value_sums.values()]
energy_cons_sums = [item['energy_consumption_returns_sum_mean'] for item in value_sums.values()]

# Plotting
fig, ax = plt.subplots()
bar_width = 0.35
index = range(len(directories))

bar1 = ax.bar(index, running_speed_sums, bar_width, label='Running Speed')
bar2 = ax.bar(index, energy_cons_sums, bar_width, label='Energy Consumption', bottom=running_speed_sums)

ax.set_xlabel('Weights')
ax.set_ylabel('Mean sums')
ax.set_title('Mean sums of Running Speed and Energy Consumption for Each Weight')
ax.set_xticks(index)
ax.set_xticklabels(directories)
ax.legend()

plt.show()