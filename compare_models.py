import csv
from colorsys import hls_to_rgb
import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.graph_objects as go

### CANNOT BE USED WITH OLD CSV FILES ###
### NEW VERSION ###

path='/home/oskar/Thesis/model_comparison_results' # paths need to be correct
newline=''


def get_distinct_colors(n):

    colors = []

    for i in np.arange(0., 360., 360. / n):
        h = i / 360.
        l = (50 + np.random.rand() * 10) / 100.
        s = (90 + np.random.rand() * 10) / 100.
        colors.append(hls_to_rgb(h, l, s))

    return colors


def convert_key_to_tuple(key):
    #return a tuple of values based on which the keys are sorted
    key_values = key.split('_')
    key_values = [value for value in key_values if value] 
    return tuple(map(float, key_values))


def sort_dictionaries(path):
    #Set up dictionaries
    value_sums = {}
    value_sums_mean = {}
    link_lengths = {}
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
                # Go through and save data to dictionaries
                if os.path.isdir(directorypath2):
                    total_run_spd_reward = np.array([]) #reset when in new file
                    total_energy_cons_reward = np.array([])
                    link_lenghts_ind = np.array([])
                    for filename in os.listdir(directorypath2):
                        if filename.endswith(".csv"):
                            filepath = os.path.join(directorypath2, filename)
                            with open(filepath, newline=newline) as file:
                                reader = csv.reader(file)
                                rows = [] # list for read values
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
                                #values to dict
                                value_sums_mean[directoryname][directory_keyname] = {'running_speed_returns_sum_mean':np.mean(total_run_spd_reward), 'energy_consumption_returns_sum_mean':np.mean(total_energy_cons_reward)}
                                link_lengths[directoryname][directory_keyname] = link_lenghts_ind
                                value_sums[directoryname][directory_keyname] = {'running_speed_returns_sum': total_run_spd_reward , 'energy_consumption_returns_sum': total_energy_cons_reward}
    return value_sums, value_sums_mean, link_lengths


if __name__ == "__main__":
    
    #the main is becomming awfully long, maybe these should just be put to functions....
    
    value_sums, value_sums_mean, link_lengths = sort_dictionaries(path)
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


    #Calculate values to arrays
    reward_sums = np.array([(sorted_mean_value_sums[key1][key2]['running_speed_returns_sum_mean'], 
                                sorted_mean_value_sums[key1][key2]['energy_consumption_returns_sum_mean']) for key1 in sorted_mean_value_sums.keys() for key2 in sorted_mean_value_sums[key1].keys()])

    value_std = np.array([[np.std(sorted_value_sums[key1][key2]['running_speed_returns_sum'], axis=0),
                        np.std(sorted_value_sums[key1][key2]['energy_consumption_returns_sum'], axis=0)]
                        for key1 in sorted_mean_value_sums.keys() for key2 in sorted_mean_value_sums[key1].keys()])

    link_lengths_array = np.array([[weights[key] for key in weights] for weights in sorted_link_lengths.values()])
    link_lengths_mean_array = np.array([np.mean(list(weights.values()), axis=0) for weights in sorted_link_lengths.values()])
    link_lengths_std_array = np.array([np.std(list(weights.values()), axis=0) for weights in sorted_link_lengths.values()])
    
    
    ######bar plot######
    fig, ax = plt.subplots()
    bar_width = 0.3
    off_set = 0.15
    group_offset = 0.5
    index = np.arange(len(reward_sums))

    bar1 = ax.bar(index - off_set, reward_sums[:, 0], bar_width, label='Running Speed')
    bar2 = ax.bar(index + off_set, reward_sums[:, 1], bar_width, label='Energy Consumption')

    ax.errorbar(index - off_set, [sorted_mean_value_sums[key1][key2]['running_speed_returns_sum_mean']
                                for key1 in sorted_mean_value_sums.keys() for key2 in sorted_mean_value_sums[key1].keys()],
                yerr=[value_std[i][0] for i in range(len(value_std))], fmt='none', color='black', capsize=5)

    ax.errorbar(index + off_set, [sorted_mean_value_sums[key1][key2]['energy_consumption_returns_sum_mean']
                                for key1 in sorted_mean_value_sums.keys() for key2 in sorted_mean_value_sums[key1].keys()],
                yerr=[value_std[i][1] for i in range(len(value_std))], fmt='none', color='black', capsize=5)

    ax.set_xlabel('Weights')
    ax.set_ylabel('Mean sums')
    ax.set_title('Mean sums of Running Speed and Energy Consumption for Each Weight')
    ax.set_xticks(index)
    ax.set_xticklabels([key2 for key1 in sorted_mean_value_sums.keys() for key2 in sorted_mean_value_sums[key1].keys()], rotation=45, ha='right')
    ax.legend()
    
    
    #####scatter plot#####
    fig2, ax2 = plt.subplots()
    ax2.set_ylabel('Energy')
    ax2.set_xlabel('Speed')
    ax2.set_title('Mean sums of Running Speed and Energy Consumption for Each Weight')

    unique_weight_groups = sorted(set(sorted_mean_value_sums.keys()), key=convert_key_to_tuple) #sorted(set([key1 for key1 in sorted_mean_value_sums.keys()]))
    #print(unique_weight_groups)
    #color_dict = {weight_group: plt.get_cmap('magma')(i / len(unique_weight_groups)) for i, weight_group in enumerate(unique_weight_groups)}
    distinct_colors = get_distinct_colors(len(unique_weight_groups)) # works better to get colors more apart from each other
    adjusted_color_dict = {weight_group: distinct_colors[i] for i, weight_group in enumerate(unique_weight_groups)}
    legend_added = {} # keep track of added legends for weight groups

    #shapes of markers
    marker_shapes = [".", ",", "o", "v", "^", "<", ">", "p", "*", "+", "h", "D", "8", ""]

    for index, (key1, key2) in enumerate([(key1, key2) for key1 in sorted_mean_value_sums.keys() for key2 in sorted_mean_value_sums[key1].keys()]):
        mask = [key_compare == key1 for key_compare in sorted_mean_value_sums.keys()]
        weight_group = unique_weight_groups[np.where(mask)[0][0]]  # Find the index where the mask is True
        
        if weight_group not in legend_added:
            shape = marker_shapes[len(legend_added) % len(marker_shapes)]
            sc = ax2.scatter(reward_sums[index, 0], reward_sums[index, 1], s=150, color=adjusted_color_dict[weight_group], marker=shape, label=weight_group)   
            legend_added[weight_group] = True
        else:
            sc = ax2.scatter(reward_sums[index, 0], reward_sums[index, 1], s=150, color=adjusted_color_dict[weight_group], marker=shape)

    ###Add or dont add annotations per model###    
    ###annote the point to scatter plot###
    # for index, txt in enumerate([key2 for key1 in sorted_mean_value_sums.keys() for key2 in sorted_mean_value_sums[key1].keys()]):
    #    ax2.annotate(txt, (reward_sums[index, 0], reward_sums[index, 1]), textcoords="offset points", xytext=(0, 10), ha='center')

    #####Error bars - add or dont add#####
    # ax2.errorbar(reward_sums[:, 0], reward_sums[:, 1],
    #     xerr=[value_std[i][0] for i in range(len(value_std))],
    #     yerr=[value_std[i][1] for i in range(len(value_std))],
    #     fmt='_', capsize=5, color='black', label='Error bars')
    ax2.legend()
    
    ##### link length plots #####
    
    #####OPTION 1 BARPLOT ######

    #weight_categories = list(sorted_link_lengths.keys())

    # bar_width = 0.15
    # group_offset = 0.4
    #seed_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Add more colors as needed

    # for i, weight_category in enumerate(weight_categories):
    #     fig, ax3 = plt.subplots(figsize=(12, 6))
    #     index = np.arange(link_lengths_array.shape[2])

    #     for j in range(link_lengths_array.shape[1]):
    #         ax3.bar(index + j * bar_width, link_lengths_array[i, j, :], bar_width, label=f'Seed_{j}', color=seed_colors[j])

    #     #ax3.plot(index + group_offset, link_lengths_mean_array[i], color='orange', marker='o', label='Mean')
    #     ax3.errorbar(index + group_offset, link_lengths_mean_array[i], yerr=link_lengths_std_array[i], color='orange', marker='o', linestyle='-', linewidth=2, label='Error Bar')

    #     ax3.set_ylabel('Link Length')
    #     ax3.set_xlabel('Link Index')
    #     ax3.set_title(f'Comparison of Link Lengths and Mean Link Lengths ({weight_category})')
    #     ax3.legend()

    # plt.tight_layout()
    
    #####OPTION 2 PLOTLY ######
    
    ###### PLOT MEAN ######
    
    weight_categories = list(sorted_link_lengths.keys())
    #print(weight_categories)
    distinct_fig_colors = get_distinct_colors(link_lengths_array.shape[1]) # bar plot color
    #distinct_error_colors = get_distinct_colors(link_lengths_array.shape[0])
    distinct_error_colors = list(plt.get_cmap('viridis_r')(i / len(weight_categories)) for i, _ in enumerate(weight_categories))
    
    fig1 = go.Figure()
    # Add bar plots for each weight category
    for i, weight_category in enumerate(weight_categories):
        index = np.arange(link_lengths_array.shape[2])

        ## MEAN AND STD PLOT
        color = f'rgb({distinct_error_colors[i][0]*255},{distinct_error_colors[i][1]*255},{distinct_error_colors[i][2]*255})'
        fig1.add_trace(go.Scatter(
            x=index + group_offset,
            y=link_lengths_mean_array[i],
            mode='markers+lines',
            name=weight_categories[i],#'Mean',
            marker=dict(color=color),#dict(color='orange'),
            error_y=dict(
                type='data',
                array=link_lengths_std_array[i],
                visible=True,
                color=color#'orange'
            )
        ))
        
    fig1.update_layout(
        barmode='group',
        xaxis=dict(title='Link Index'),
        yaxis=dict(title='Link Length'),
        title='Comparison of Mean Link Lengths',#'Comparison of Mean Link Lengths',
        showlegend=True
    )
    fig1.show()

    ###### PLOT ALL ######
    figplo = go.Figure()
    # Add bar plots for each weight category
    for i, weight_category in enumerate(weight_categories):
        index = np.arange(link_lengths_array.shape[2])
        # barplots
        # for j in range(link_lengths_array.shape[1]):
        #     figplo.add_trace(go.Bar(
        #         x=index + j * bar_width,
        #         y=link_lengths_array[i, j, :],
        #         name=f'Seed_{j}',
        #         marker_color=distinct_fig_colors[j]  # Use marker_color instead of marker=dict(color=...)
        #     ))

        ## MEAN AND STD PLOT
        # color = f'rgb({distinct_error_colors[i][0]*255},{distinct_error_colors[i][1]*255},{distinct_error_colors[i][2]*255})'
        # figplo.add_trace(go.Scatter(
        #     x=index + group_offset,
        #     y=link_lengths_mean_array[i],
        #     mode='markers+lines',
        #     name=weight_categories[i],#'Mean',
        #     marker=dict(color=color),#dict(color='orange'),
        #     error_y=dict(
        #         type='data',
        #         array=link_lengths_std_array[i],
        #         visible=True,
        #         color=color#'orange'
        #     )
        # ))
        
        # INDIVIDUAL SEEDS PLOTTED
        color = f'rgb({distinct_error_colors[i][0]*255},{distinct_error_colors[i][1]*255},{distinct_error_colors[i][2]*255})'
        for j in range(link_lengths_array.shape[1]):
            figplo.add_trace(go.Scatter(
                x=index + group_offset,
                y=link_lengths_array[i, j, :],
                mode='markers+lines',
                name=str(weight_categories[i])+"_"+str(j+1),#'Mean',
                marker=dict(color=color),#dict(color='orange'),
                # error_y=dict(
                #     type='data',
                #     array=link_lengths_std_array[i],
                #     visible=True,
                #     color=color#'orange'
                # )
            ))
        
    figplo.update_layout(
        barmode='group',
        xaxis=dict(title='Link Index'),
        yaxis=dict(title='Link Length'),
        title='Comparison of Link Lengths',#'Comparison of Mean Link Lengths',
        showlegend=True
    )

    ##### PLOT IND LINK LENGTHS #####
    
    figplo.show()
    weight_categories = list(sorted_link_lengths.keys())
    distinct_error_colors = list(plt.get_cmap('viridis_r')(i / len(weight_categories)) for i, _ in enumerate(weight_categories))

    for j in range(link_lengths_array.shape[2]):
        fig = go.Figure()
        index = np.arange(link_lengths_array.shape[0])+1
        xticklabels = np.arange(link_lengths_array.shape[1])+1#sorted(weight_categories, key=lambda x: tuple(map(float, x.split('_'))))
        
        for i, weight_category in enumerate(weight_categories):
            color = f'rgb({distinct_error_colors[i][0]*255},{distinct_error_colors[i][1]*255},{distinct_error_colors[i][2]*255})'
            #print(i)
            fig.add_trace(go.Scatter(
                    x=index + group_offset,
                    y=link_lengths_array[i, :, j],
                    mode='markers+lines',
                    name=f'{weight_category} Link {j + 1}',
                    marker=dict(color=color),
                    line=dict(color=color)
            ))
        
        fig.update_layout(
            xaxis=dict(title='Seed', tickmode='array', tickvals=index + group_offset, ticktext=xticklabels),
            yaxis=dict(title='Link Length'),
            title=f'Comparison of Link Lengths for Link {j + 1}',
            showlegend=True
        )
        fig.show()
    
    
    plt.show()
