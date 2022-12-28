# Wave-Defense-Baselines


This repository contains the source code for training different baselines for the WaveDefense learning environment.


### Running the baselines (online RL)

For running PPO on the image-based version of WaveDefense run (note that wandb logging is set to default, you can disable it by removing the last 2 flags):

python src/models/ppo_image.py\ 
--num-envs 16\
--num-steps 2048\ 
--num-minibatches 64\ 
--update-epochs 10\
--clip-coef 0.2\
--ent-coef 0\
--total-timesteps 10000000\
--track\
--wandb-project-name WaveDefense\
--eval-dir checkpoints_img

For running PPO on the tabular-based environment run:

python src/models/ppo_tabular.py --num-envs 16 --num-steps 2048 --num-minibatches 64 --update-epochs 10 --clip-coef 0.2 --ent-coef 0 --total-timesteps 10000000 --track --wandb-project-name WaveDefense --eval-dir checkpoints_tabular


For running PPO on image-based environment **with no reward** using Inverse Curiosoty Module:

python src/models/ppo_icm.py --num-envs 16 --num-steps 2048 --num-minibatches 64 --update-epochs 10 --clip-coef 0.2 --ent-coef 0 --total-timesteps 10000000 --track --wandb-project-name WaveDefense --eval-dir checkpoints_icm 

### Running the baselines (offline RL)

1. Create rollouts with a random policy and store tuples of experience: (observation, action, reward, next_observation, done)

python3 src/create_trajectories.py

2. To train Implicit Q-learning on the collected data:

python3 src/models/IQL/train.py

### Headless mode   

For running in machines with no display (e.g. compute cluster) use:

os.environ["SDL_VIDEODRIVER"] = "dummy"

Note that the latter will disable rendering

### Warning
UserWarning: WARN: The environment WaveDefense-v0 is out of date. You should consider upgrading to version `v1` with the environment ID `WaveDefense-v1`.