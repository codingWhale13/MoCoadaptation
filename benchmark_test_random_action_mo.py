
import gym
import numpy as np
#from gym import wrappers
from Environments import evoenvsMO as envsmo

#env = BasicWrapper(gym.make('CartPole-v0'))

#env = gym.make(envsmo.HalfCheetahEnv)

#env.reset()
#env = gym.make('CartPole-v0')
#env.reset()

#env = gym.Wrapper(gym.make("CartPole-v0")) 
env = gym.Wrapper(envsmo.HalfCheetahEnvMO()) #gym.Wrapper(gym.make("HalfCheetah"))

for i_episode in range(100):
    observation = env.reset()
    #while (True):
    for _ in range(1, 1000):
        #env.render()
        action = env.action_space.sample()
        #action = np.random.randint(env.action_space.n, size=num)
        #action = env.
        observation, reward, done, *info = env.step(action)
        print("Episode finished after {} episodes".format(i_episode+1))
        print(f"obvervation: {observation}, reward: {reward}, done: {done}, *info: {info}")
        if done:
            print("Episode finished after {} episodes".format(i_episode+1))
            print(f"obvervation: {observation}, reward: {reward}, done: {done}, *info: {info}")
            break

env.close()