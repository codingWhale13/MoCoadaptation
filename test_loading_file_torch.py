import torch
import pickle

if __name__ == "__main__":

    try:
        #loaded_object = torch.load('/home/oskar/Thesis/Fri_Sep_29_10:50:16_2023__5ca370bd/checkpoints/checkpoint_design_1.chk')#'/home/oskar/Thesis/checkpoint_design_60/data.pkl')
        loaded_object = torch.load('/home/oskar/Thesis/Results_scalarized/results_with_wandb/Wed_Oct_18_18:29:14_2023__49c4bc4c/checkpoints/checkpoint_design_1.chk') #, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"TORCH: An error occurred: {e}")
        
    print(loaded_object)

    #try:
    # with open('/home/oskar/Thesis/Fri_Sep_29_10:50:16_2023__5ca370bd/checkpoints/checkpoint_design_15.chk', 'rb') as file:
    #     #with open('/home/oskar/Thesis/Fri_Sep_29_10:50:16_2023__5ca370bd/checkpoints/checkpoint_design_15.chk/data.pkl', 'rb') as file:
    #     loaded_content = pickle.load(file)
    # print(loaded_content)
    #except Exception as e:
    #    print(f"PICKLE: An error occurred: {e}")