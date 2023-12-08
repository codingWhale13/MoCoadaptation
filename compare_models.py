import csv
import matplotlib.pyplot as plt
import numpy as np
import os

path='/home/oskar/Thesis/model_comparison_results'
path_link='/home/oskar/Thesis/Model_scalarized/results_with_rescaling/random seed/' # remember to change the weights -> /0.0_1.0 etc...
newline=''

value_sums = {}
value_sums_mean = {}
link_lenghts = {}




def check_path(path_link, weightdir):
    #return the correct path with correct directory per weight
    path_dir = os.path.join(path_link, weightdir)
    return path_dir

def read_morphology(morphologydirname, weightdir, morphology_number) -> list:
    
    """ Returns a list of values read from cvs file per row
    
    Returns:
        a list containing csv file values
    """    
    rows = []
    path_dir = check_path(path_link, weightdir)
    morphology_number = morphology_number  + ".csv"
    for dirname in os.listdir(path_dir):
        if dirname == morphologydirname:
            filepath = os.path.join(path_dir, dirname)
            for filename in os.listdir(filepath):
                if filename.endswith(morphology_number):
                    filename2 = os.path.join(filepath, filename)
                    with open(filename2, newline=newline) as file:
                        reader = csv.reader(file)
                        for row in reader:
                            rows.append(row)
    return rows

for directoryname in os.listdir(path):
    directorypath = os.path.join(path, directoryname)
    value_sums_mean[directoryname] = {}
    value_sums[directoryname] = {}
    if os.path.isdir(directorypath):
        for directoryname2 in os.listdir(directorypath):
            morphologydir = directoryname2[:-2]
            # since link lenghts are saved in different files, we will need to know the weights to access the correct directory
            weightdir = morphologydir.split('[')[-1]
            weightdir = weightdir[:-1]
            weightdir = weightdir.replace(", ", "_")
            # check for morphology number since one iteration has 59, its hardcoded value here, otherwise we always just want the 60 design
            if morphologydir == 'Tue_Dec__5_19:07:12_2023__c7d34aee[0.0, 1.0]':
                num = '59'
            else:
                num = '60'
            model_file = read_morphology(morphologydir, weightdir, num) # read model csv file as list
            link_lengths = np.array(model_file[1], dtype=float) # index link lengths from the file
            link_lenghts[directoryname2] = link_lengths 
            directorypath2 = os.path.join(directorypath, directoryname2)
            
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
#print(value_sums_mean['0.8_0.2'])
#print(link_lenghts)    
directories = list(sorted(link_lenghts.keys()))
#print(directories)

keys = value_sums_mean.keys()
print(keys)
 
value_sums_mean_sorted = dict(sorted(value_sums_mean.items())) # sort keys  
print(value_sums_mean_sorted)

#directories = list(sorted(value_sums_mean.keys())) # ORIG
#print(directories)
key_order = [key for key in value_sums_mean_sorted]

#scaled
#key_order = ['0.0_1.0', '0.01_0.99'] + [key for key in value_sums_mean_sorted if key not in ['0.0_1.0', '0.01_0.99', '1.0_0.0', '0.99_0.01']] + ['0.99_0.01', '1.0_0.0'] # switch places or 0.0_1.0 and 0.01_0.99
#unscaled
#key_order = ['0.0_1.0'] + [key for key in value_sums_mean_sorted if key not in ['0.0_1.0', '1.0_0.0']] + ['1.0_0.0'] # switch places or 0.0_1.0 and 0.01_0.99
#print(key_order)
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
ax2.errorbar(running_speed_sums, energy_cons_sums,
            xerr=[value_std[index][0] for index in key_order],
            yerr=[value_std[index][1] for index in key_order],
            fmt=':b',label="Bar plot")
ax2.legend()

plt.show()