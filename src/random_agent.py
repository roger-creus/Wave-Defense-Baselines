import os
import WaveDefense
import gym


env = gym.make("WaveDefense-v0")                # for image-based state representations
#env = gym.make("WaveDefenseNoReward-v0")         # for image-based state representations and no reward distribution
#env = gym.make("WaveDefense-v1")                # for tabular state representations

# seeding
env.seed(10)

env.reset()
done = False

steps = 0

while done is False:
    steps += 1
    obs, rew, done, info = env.step(env.action_space.sample()) # take a random action

print(steps)
env.close()