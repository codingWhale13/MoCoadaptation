import csv
import matplotlib.pyplot as plt
import numpy as np
import os


path='/home/oskar/Work/Changes/MoCoadaptation/data_exp_sac_pso_sim/Wed_Nov_15_11:56:27_2023__aa88d15c'
#'/home/oskar/Thesis/episodic_rewards/'
newline=''
total_run_spd_reward = np.array([])
total_energy_cons_reward = np.array([])

for filename in os.listdir(path):
    if filename.endswith(".csv"):
        filepath = os.path.join(path, filename)
        running_speed_reward = np.array([])
        energy_consumption_reward = np.array([])
        with open(filepath, newline=newline) as file:
            reader = csv.reader(file)
            for row in reader:
                running_speed_reward = np.append(running_speed_reward, float(row[0]))
                energy_consumption_reward = np.append(energy_consumption_reward, float(row[1]))
                
            run_speed_sum_reward_sum = np.sum(running_speed_reward)
            energy_cons_reward_sum = np.sum(energy_consumption_reward)
            #mean_running_speed_reward = np.mean(running_speed_reward)
            #mean_energy_consumption_reward = np.mean(energy_consumption_reward)
            #total_run_spd_reward = np.append(total_run_spd_reward, mean_running_speed_reward)
            #total_energy_cons_reward= np.append(total_energy_cons_reward, mean_energy_consumption_reward)
            total_run_spd_reward = np.append(total_run_spd_reward, run_speed_sum_reward_sum)
            total_energy_cons_reward= np.append(total_energy_cons_reward, energy_cons_reward_sum)

#print("Mean rewards:")  
print("Sum of rewards")
print(total_run_spd_reward)  
print(total_energy_cons_reward)         
#print(total_run_spd_reward)
#print(total_energy_cons_reward)
print(f"Mean of the runs of running speed rewards: {np.mean(total_run_spd_reward)}")
print(f"Mean of the runs of energy consumption rewards: {np.mean(total_energy_cons_reward)}")
print(f"Ratio of means: {np.mean(total_run_spd_reward)/np.mean(total_energy_cons_reward)}")

# Plotting
plt.plot(total_run_spd_reward, label='Running Speed Reward')
plt.plot(total_energy_cons_reward, label='Energy Consumption Reward')
plt.xlabel('Run Index')
plt.ylabel('Mean Reward')
plt.title('Mean Rewards Across Runs')
plt.legend()
plt.show()
  