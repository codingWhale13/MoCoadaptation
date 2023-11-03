import torch
import pickle

if __name__ == "__main__":

    try:
        loaded_object = torch.load('/home/oskar/Thesis/Fri_Sep_29_10:50:16_2023__5ca370bd/checkpoints/checkpoint_design_15.chk')#'/home/oskar/Thesis/checkpoint_design_60/data.pkl')
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