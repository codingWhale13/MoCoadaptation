import numpy as np

dic1 = {'obj': np.array([2, 3])}

print(dic1['obj'][1])
print(dic1)

numpy_dict1 = np.array(dic1)

print(numpy_dict1)

numpy_tuple = np.array((dic1['obj']))

print(numpy_tuple)

arr1 = np.array([1, 2, 3])

print(arr1)

reward_episode = []

i = 0
for i in range(1, 6):

    reward = np.array([i+i, -i]) * 1.2
    reward_episode.append(reward)
    
print(reward_episode[4])
print(reward_episode[0][1])
print(reward_episode)
m1, m2 = reward_episode[4]

print(f"m1 : {m1} and m2 : {m2}")

# rewards = np.array([])

# for i in range(1, 5):

#     reward = np.array([i, -i]) * 1.2
#     np.concatenate(rewards, reward)
    
# print(rewards)


r2= np.array([0, 0])
r3 = np.array([3, 3])

r1 = np.array([5, 4])

r2 += r1

r3 += r2


print(r2)
print(r3)

ep1 = np.array([10, 5])

reward_mean = np.mean(ep1)

print(reward_mean)

print(np.column_stack((r1, [2, 3])))