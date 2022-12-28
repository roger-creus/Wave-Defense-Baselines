import torch
import numpy as np
import os
from IPython import embed
import gym
import wandb

def load_dataset(path):
    print("Loading data...")

    # data types are obs, actions, rewards, next_obs and dones
    data_types = os.listdir(path)

    obs = []
    next_obs = []
    actions = []
    rewards = []
    dones = []
    
    for d in data_types:
        # get data type path
        data_type_path = os.path.join(path, d)

        print("loading " + d)
        
        # we collected data from multiple envs with different seed
        envs = os.listdir(data_type_path)

        for e in envs:
            env_path = os.path.join(data_type_path, e)

            for traj in sorted(os.listdir(env_path)):
                traj_path = os.path.join(env_path, traj)

                traj = np.load(str(traj_path), allow_pickle=True)

                if d == "actions":
                    actions.append(traj.flatten())
                elif d == "obs":
                    obs.append(traj.flatten())
                elif d == "next_obs":
                    next_obs.append(traj.flatten())
                elif d == "dones":
                    dones.append(traj.flatten())
                elif d == "rewards":
                    rewards.append(traj.flatten())

    obs = np.concatenate(np.array(obs, dtype='object'))
    next_obs = np.concatenate(np.array(next_obs, dtype='object'))
    dones = np.concatenate(np.array(dones, dtype='object'))
    rewards = np.concatenate(np.array(rewards, dtype='object'))
    actions = np.concatenate(np.array(actions, dtype='object'))

    obs = obs.reshape(-1, 84, 84, 3)
    next_obs = next_obs.reshape(-1, 84, 84, 3) 
    
    # IF NEEDED TO BE NORMALIZED BETWEEN 0 AND 1!
    obs = obs / 255.
    next_obs = next_obs / 255.

    print(obs.shape)

    return obs, actions, rewards, next_obs, dones


def make_env(env_id, seed):
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    if env_id == "WaveDefense-v0" or env_id == "WaveDefenseNoReward-v0":
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        #env = gym.wrappers.GrayScaleObservation(env)
        #env = gym.wrappers.FrameStack(env, 4)
        print("--------- Training on the image-based environment ---------")
    else:
        print("--------- Training on the tabular-based environment ---------")

    # seeding
    env.seed(seed)        
    return env

def save(args, save_path, save_name, model, wandb, ep=None):
    import os
    save_dir = os.path.join(save_path, "/trained_models/") 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model.state_dict(), os.path.join(save_dir,  args.run_name + save_name + str(ep) + ".pth"))
        wandb.save( os.path.join(save_dir,  args.run_name + save_name + str(ep) + ".pth"))
    else:
        torch.save(model.state_dict(),  os.path.join(save_dir,  args.run_name + save_name + str(ep) + ".pth"))
        wandb.save( os.path.join(save_dir,  args.run_name + save_name + str(ep) + ".pth"))


class VideoWrapper(gym.Wrapper):
    """Gathers up the frames from an episode and allows to upload them to Weights & Biases
    Thanks to @cyrilibrahim for this snippet
    """

    def __init__(self, env, update_freq=25):
        super(VideoWrapper, self).__init__(env)
        self.episode_images = []
        # we need to store the last episode's frames because by the time we
        # wanna upload them, reset() has juuust been called, so the self.episode_rewards buffer would be empty
        self.last_frames = None

        # we also only render every 20th episode to save framerate
        self.episode_no = 0
        self.render_every_n_episodes = update_freq  # can be modified

    def reset(self, **kwargs):
        self.episode_no += 1
        if self.episode_no == self.render_every_n_episodes:
            self.episode_no = 0
            self.last_frames = self.episode_images[:]
            self.episode_images.clear()

        state = self.env.reset()

        return state

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        if self.episode_no + 1 == self.render_every_n_episodes:
            frame = np.copy(self.env.render())
            self.episode_images.append(frame)

        return state, reward, done, info

    def send_wandb_video(self):
        if self.last_frames is None or len(self.last_frames) == 0:
            print("Not enough images for GIF. continuing...")
            return
        lf = np.array(self.last_frames)
        print(lf.shape)
        frames = np.swapaxes(lf, 1, 3)
        frames = np.swapaxes(frames, 2, 3)
        wandb.log({"video": wandb.Video(frames, fps=60, format="gif")})
        print("=== Logged GIF")

    def get_wandb_video(self):
        if self.last_frames is None or len(self.last_frames) == 0:
            print("Not enough images for GIF. continuing...")
            return None
        lf = np.array(self.last_frames)
        print(lf.shape)
        frames = np.swapaxes(lf, 1, 3)
        frames = np.swapaxes(frames, 2, 3)
        return frames