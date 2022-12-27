import gym
import argparse
import numpy as np
import os
import random
import cv2
import matplotlib.pyplot as plt
import WaveDefense

from IPython import embed

# for running in a cluster with no display
os.environ["SDL_VIDEODRIVER"] = "dummy"

parser = argparse.ArgumentParser()
parser.add_argument('--traj-length', type=int, default = 500,
                    help='trajectory length')
parser.add_argument('--traj-num', type=int, default=500,
                    help='num of trajectories')
parser.add_argument('--envs-num', type=int, default=10,
                    help='num of different environments')            
parser.add_argument('--save-path', type=str, default="src/representations/trajectories/",
                    help='save path')
parser.add_argument('--seed', type=int, default=1,
                    help='seed of the environment')

args = parser.parse_args()

traj_len = args.traj_length
traj_num = args.traj_num
envs_num = args.envs_num

for env_ in range(envs_num):
    env = gym.make("WaveDefense-v0")
    env.seed(args.seed + env_)

    envs_path = "env_" + str(args.seed + env_) + "/"
    envs_obs_path = args.save_path + "observations/" + envs_path
    envs_actions_path = args.save_path + "actions/" + envs_path

    if not os.path.exists(envs_obs_path):
        os.makedirs(envs_obs_path)
    
    if not os.path.exists(envs_actions_path):
        os.makedirs(envs_actions_path)
    
    print("creating trajectories with seed: " + str(args.seed + env_))

    for i in range(traj_num):
        steps = 0
        done = False
        obs = env.reset()
        
        trajectory = []
        trajectory_actions = []

        while steps < traj_len and not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

            # img to gray scale, eventually the model uses gray images
            #obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
            #obs = obs[None,:, :]
            
            # scale imgs to [0,1], the model uses this format
            #obs = np.array(obs).astype(np.float32) / 255.0
            
            trajectory.append(obs)
            trajectory_actions.append(action)

        trajectory = np.array(trajectory)
        trajectory_actions = np.array(trajectory_actions)
        
        trajectory_observations_path = envs_obs_path + "trajectory_observations_"  + str(i) + ".npy"
        trajectory_actions_path = envs_actions_path +  "trajectory_actions_" + str(i) + ".npy"

        with open(trajectory_observations_path, 'wb') as tf:
            np.save(tf, trajectory)
        
        with open(trajectory_actions_path, 'wb') as ta:
            np.save(ta, trajectory_actions)