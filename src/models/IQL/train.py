import gym
import numpy as np
from collections import deque
import torch
import wandb
import argparse
import glob
import random
from torch.utils.data import DataLoader, TensorDataset
import WaveDefense
import os
from IPython import embed

from agent import IQL
from utils import load_dataset, make_env, save, VideoWrapper

# for running in a cluster with no display
os.environ["SDL_VIDEODRIVER"] = "dummy"

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--data_path", type=str, default="/home/roger/Desktop/Wave-Defense-Baselines-/src/representations/trajectories", help="Where the collected experiences are")
    parser.add_argument("--run_name", type=str, default="IQL", help="Run name, default: SAC")
    parser.add_argument("--env", type=str, default="WaveDefense-v0", help="Gym environment name, default: Pendulum-v0")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes, default: 100")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--save_every", type=int, default=25, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--save_path", type=str, default="/home/roger/Desktop/Wave-Defense-Baselines-/checkpoints", help="Where to save the model")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size, default: 256")
    parser.add_argument("--hidden_size", type=int, default=256, help="")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="")
    parser.add_argument("--temperature", type=float, default=3, help="")
    parser.add_argument("--expectile", type=float, default=0.8, help="")
    parser.add_argument("--tau", type=float, default=5e-3, help="")
    parser.add_argument("--eval_every", type=int, default=1, help="")
    
    args = parser.parse_args()
    return args

def prep_dataloader(path, env_id="WaveDefense-v0", batch_size=256, seed=1):
    obs, actions, rewards, next_obs, dones = load_dataset(path)

    tensordata = TensorDataset(torch.from_numpy(np.array(obs, dtype=np.float16)).permute(0,3,1,2).float(),
                               torch.from_numpy(np.array(actions, dtype=np.float16)).unsqueeze(1).float(),
                               torch.from_numpy(np.array(rewards, dtype=np.float16)).unsqueeze(1).float(),
                               torch.from_numpy(np.array(next_obs, dtype=np.float16)).permute(0,3,1,2).float(),
                               torch.from_numpy(np.array(dones, dtype=np.float16)).unsqueeze(1).float())

    dataloader  = DataLoader(tensordata, batch_size=batch_size, shuffle=True)
    
    return dataloader

def evaluate(policy, eval_runs=5): 
    """
    Makes an evaluation run with the current policy
    """
    env = make_env(config.env, np.random.randint(1000))
    env = VideoWrapper(env, update_freq = 3)

    reward_batch = []
    for i in range(eval_runs):
        state = env.reset()

        rewards = 0
        while True:
            action = policy.get_action(state, eval=True)

            state, reward, done, _ = env.step(action)
            rewards += reward
            if done:
                break
        reward_batch.append(rewards)

    env.send_wandb_video()
    env.close()

    return np.mean(reward_batch)

def train(config):
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    dataloader = prep_dataloader(env_id=config.env, batch_size=config.batch_size, seed=config.seed, path=config.data_path)

    env = make_env(config.env, config.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    batches = 0
    average10 = deque(maxlen=10)
    
    with wandb.init(project="WaveDefense", name=config.run_name, config=config):
        
        agent = IQL(state_size=env.observation_space.shape,
                    action_size=env.action_space.n,
                    learning_rate=config.learning_rate,
                    hidden_size=config.hidden_size,
                    tau=config.tau,
                    temperature=config.temperature,
                    expectile=config.expectile,
                    device=device)

        wandb.watch(agent, log="gradients", log_freq=10)
        
        eval_reward = evaluate(agent)
        
        wandb.log({"Test Reward": eval_reward, "Episode": 0, "Batches": batches}, step=batches)
        
        for i in range(1, config.episodes+1):
            for batch_idx, experience in enumerate(dataloader):
                states, actions, rewards, next_states, dones = experience
                states = states.to(device)
                actions = actions.to(device)
                rewards = rewards.to(device)
                next_states = next_states.to(device)
                dones = dones.to(device)
                policy_loss, critic1_loss, critic2_loss, value_loss = agent.learn((states, actions, rewards, next_states, dones))
                batches += 1

            if i % config.eval_every == 0:
                eval_reward = evaluate(agent)
                wandb.log({"Test Reward": eval_reward, "Episode": i, "Batches": batches}, step=batches)

                average10.append(eval_reward)
                print("Episode: {} | Reward: {} | Polciy Loss: {} | Batches: {}".format(i, eval_reward, policy_loss, batches,))
            
            wandb.log({
                       "Average10": np.mean(average10),
                       "Policy Loss": policy_loss,
                       "Value Loss": value_loss,
                       "Critic 1 Loss": critic1_loss,
                       "Critic 2 Loss": critic2_loss,
                       "Batches": batches,
                       "Episode": i})

            if i % config.save_every == 0:
                save(config, save_name="IQL", model=agent.actor_local, wandb=wandb, ep=0, save_path=config.save_path)
                print("saved model")

if __name__ == "__main__":
    config = get_config()
    train(config)