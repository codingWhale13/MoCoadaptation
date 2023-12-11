import csv
import matplotlib.pyplot as plt
import numpy as np
import os

path='/home/oskar/Thesis/model_comparison_results' # paths need to be correct
path_link='/home/oskar/Thesis/Model_scalarized/results_with_rescaling/random seed/' # remember to set the directories correctly
newline=''

value_sums = {}
value_sums_mean = {}
link_lenghts = {}

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
    value_sums_mean[directoryname] = {}
    value_sums[directoryname] = {}
    if os.path.isdir(directorypath):
        for directoryname2 in os.listdir(directorypath):
            #Set new directory path
            directorypath2 = os.path.join(directorypath, directoryname2)
            #print(f"directory path 2: {directorypath2}")
            
            morphologydir = directoryname2[:-2]
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
            link_lenghts[directoryname2] = link_lengths_ind 
            
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

# #scaled
# key_order = ['0.0_1.0', '0.01_0.99'] + [key for key in value_sums_mean_sorted if key not in ['0.0_1.0', '0.01_0.99', '1.0_0.0', '0.99_0.01']] + ['0.99_0.01', '1.0_0.0'] # switch places or 0.0_1.0 and 0.01_0.99
# #unscaled
# key_order = ['0.0_1.0'] + [key for key in value_sums_mean_sorted if key not in ['0.0_1.0', '1.0_0.0']] + ['1.0_0.0'] # switch places or 0.0_1.0 and 0.01_0.99
# print(key_order)

print(f" Links of the models: {link_lenghts}")

key_order_weights = list(sorted(value_sums_mean.keys()))

for weight in range(len(key_order_weights)):
    
    #print(key_order_weights[weight])
    
    #values per iteration, each iteration goes through another weight
    key_order_runs = [key for key in value_sums_mean[key_order_weights[weight]]]

    running_speed_sums = [value_sums_mean[key_order_weights[weight]][key]['running_speed_returns_sum_mean'] for key in key_order_runs]
    energy_cons_sums = [value_sums_mean[key_order_weights[weight]][key]['energy_consumption_returns_sum_mean'] for key in key_order_runs]


    value_std = {index: [np.std(value_sums[key_order_weights[weight]][index]['running_speed_returns_sum'], axis=0),
                        np.std(value_sums[key_order_weights[weight]][index]['energy_consumption_returns_sum'], axis=0)]
                for index in key_order_runs}

    #Plotting

    fig, ax = plt.subplots()
    bar_width = 0.3
    off_set = 0.15
    index = np.arange(len(key_order_runs))

    bar1 = ax.bar(index - off_set , running_speed_sums, bar_width, label='Running Speed')
    bar2 = ax.bar(index + off_set , energy_cons_sums, bar_width, label='Energy Consumption')

    #std error bars 
    ax.errorbar(index - off_set , [value_sums_mean[key_order_weights[weight]][index]['running_speed_returns_sum_mean'] for index in key_order_runs],
                yerr=[value_std[index][0] for index in key_order_runs], fmt='none', color='black', capsize=7)
    ax.errorbar(index + off_set , [value_sums_mean[key_order_weights[weight]][index]['energy_consumption_returns_sum_mean'] for index in key_order_runs],
                yerr=[value_std[index][1] for index in key_order_runs], fmt='none', color='black', capsize=7)

    ax.set_xlabel('Weights')
    ax.set_ylabel('Mean sums')
    ax.set_title('Mean sums of Running Speed and Energy Consumption for Each Weight')
    ax.set_xticks(index)
    ax.set_xticklabels(key_order_runs)
    ax.legend()

    fig2, ax2 = plt.subplots()
    ax2.scatter(running_speed_sums, energy_cons_sums, color='red')
    ax2.set_ylabel('Energy')
    ax2.set_xlabel('Speed')
    ax2.set_title('Mean sums of Running Speed and Energy Consumption for Each Weight')

    for index, weight in enumerate(key_order_runs):
        ax2.annotate(key_order_runs[index], (running_speed_sums[index],energy_cons_sums[index]), textcoords="offset points", xytext=(0,10), ha='center')
    ax2.errorbar(running_speed_sums, energy_cons_sums,
                xerr=[value_std[index][0] for index in key_order_runs],
                yerr=[value_std[index][1] for index in key_order_runs],
                fmt=':b',label="Bar plot")
    ax2.legend()

    plt.show()