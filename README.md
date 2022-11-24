# Wave-Defense-Baselines


This repository contains the source code for training different baselines for the WaveDefense learning environment.


### Running the baselines

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

### Headless mode   

For running in machines with no display (e.g. compute cluster) use:

os.environ["SDL_VIDEODRIVER"] = "dummy"

Note that the latter will disable rendering