import WaveDefense
import gym

#env = gym.make("WaveDefense-v0")                # for image-based state representations
#env = gym.make("WaveDefense-v1")                # for tabular state representations

env = gym.make("WaveDefenseNoReward-v0")         # for image-based state representations and no reward distribution
env.reset()
done = False

while done is False:
    env.render()
    obs, rew, done, info = env.step(env.action_space.sample()) # take a random action

env.close()