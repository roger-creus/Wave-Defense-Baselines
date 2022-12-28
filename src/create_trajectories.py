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
    envs_obs_path = os.path.join(args.save_path, "obs/" + envs_path)
    envs_next_obs_path = os.path.join(args.save_path, "next_obs/" + envs_path)
    envs_actions_path = os.path.join(args.save_path, "actions/" + envs_path)
    envs_rewards_path = os.path.join(args.save_path, "rewards/" + envs_path)
    envs_dones_path = os.path.join(args.save_path, "dones/" + envs_path)

    if not os.path.exists(envs_obs_path):
        os.makedirs(envs_obs_path)
        print("created path " + str(envs_obs_path))
    
    if not os.path.exists(envs_next_obs_path):
        os.makedirs(envs_next_obs_path)
        print("created path " + str(envs_next_obs_path))

    if not os.path.exists(envs_rewards_path):
        os.makedirs(envs_rewards_path)
        print("created path " + str(envs_rewards_path))
    
    if not os.path.exists(envs_actions_path):
        os.makedirs(envs_actions_path)
        print("created path " + str(envs_actions_path))

    if not os.path.exists(envs_dones_path):
        os.makedirs(envs_dones_path)
        print("created path " + str(envs_dones_path))

    
    print("creating trajectories with seed: " + str(args.seed + env_))

    for i in range(traj_num):
        steps = 0
        done = False
        obs = env.reset()
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)

        trajectory = []
        trajectory_next = []
        trajectory_rewards = []
        trajectory_actions = []
        trajectory_dones = []

        while steps < traj_len and not done:
            action = env.action_space.sample()
            
            trajectory.append(obs)
            
            obs, reward, done, info = env.step(action)

            # img to gray scale, eventually the model uses gray images
            #obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
            obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
            #obs = obs[None,:, :]
            
            # scale imgs to [0,1], the model uses this format
            #obs = np.array(obs).astype(np.float32) / 255.0

            trajectory_next.append(obs)
            trajectory_actions.append(action)
            trajectory_rewards.append(reward)
            trajectory_dones.append(done)
            
            steps += 1

        trajectory = np.array(trajectory)
        trajectory_next = np.array(trajectory_next)
        trajectory_rewards = np.array(trajectory_rewards)
        trajectory_dones = np.array(trajectory_dones)
        trajectory_actions = np.array(trajectory_actions)
        
        trajectory_observations_path = os.path.join(envs_obs_path, "trajectory_obs_"  + str(i) + ".npy")
        trajectory_actions_path = os.path.join(envs_actions_path, "trajectory_actions_" + str(i) + ".npy")
        trajectory_observations_next_path = os.path.join(envs_next_obs_path, "trajectory_nextobs_" + str(i) + ".npy")
        trajectory_rewards_path = os.path.join(envs_rewards_path, "trajectory_rewards_" + str(i) + ".npy")
        trajectory_dones_path = os.path.join(envs_dones_path, "trajectory_dones_" + str(i) + ".npy")

        with open(trajectory_observations_path, 'wb') as f:
            np.save(f, trajectory)
        
        with open(trajectory_observations_next_path, 'wb') as f1:
            np.save(f1, trajectory_next)

        with open(trajectory_rewards_path, 'wb') as f2:
            np.save(f2, trajectory_rewards)
        
        with open(trajectory_dones_path, 'wb') as f3:
            np.save(f3, trajectory_dones)

        with open(trajectory_actions_path, 'wb') as f4:
            np.save(f4, trajectory_actions)