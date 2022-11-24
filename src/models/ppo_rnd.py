import argparse
import os
import random
import time
from collections import deque
from distutils.util import strtobool

import WaveDefense
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gym.wrappers.normalize import RunningMeanStd
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.utils import *
from utils.networks import *
from utils.rnd import *

# for running in a cluster with no display
os.environ["SDL_VIDEODRIVER"] = "dummy"

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="WaveDefenseNoReward-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=16,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.999,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=64,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.001,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--sticky-action", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, sticky action will be used")
    parser.add_argument("--eval-dir", type=str, default="checkpoints/",
        help="where to save best models")

    # RND arguments
    parser.add_argument("--update-proportion", type=float, default=0.25,
        help="proportion of exp used for predictor update")
    parser.add_argument("--int-coef", type=float, default=1.0,
        help="coefficient of extrinsic reward")
    parser.add_argument("--ext-coef", type=float, default=0.0,
        help="coefficient of intrinsic reward")
    parser.add_argument("--int-gamma", type=float, default=0.99,
        help="Intrinsic reward discount rate")
    parser.add_argument("--num-iterations-obs-norm-init", type=int, default=50,
        help="number of iterations to initialize the observations normalization parameters")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, False, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    envs_test = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, False, run_name) for i in range(1)]
    )

    agent = Agent_RND(envs).to(device)
    rnd_model = RNDModel(1, envs.single_action_space.n).to(device)
    combined_parameters = list(agent.parameters()) + list(rnd_model.predictor.parameters())
    optimizer = optim.Adam(
        combined_parameters,
        lr=args.learning_rate,
        eps=1e-5,
    )

    reward_rms = RunningMeanStd()
    obs_rms = RunningMeanStd(shape=(1, 1, 84, 84))
    discounted_reward = RewardForwardFilter(args.int_gamma)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    curiosity_rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    ext_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    int_values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    avg_returns = deque(maxlen=20)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    print("Start to initialize observation normalization parameter.....")
    next_ob = []
    for step in range(args.num_steps * args.num_iterations_obs_norm_init):
        acs = np.random.randint(0, envs.single_action_space.n, size=(args.num_envs,))
        s, r, d, _ = envs.step(acs)
        next_ob += s[:, 3, :, :].reshape([-1, 1, 84, 84]).tolist()

        if len(next_ob) % (args.num_steps * args.num_envs) == 0:
            next_ob = np.stack(next_ob)
            obs_rms.update(next_ob)
            next_ob = []
    print("End to initialize...")

    best_reward = 0

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                value_ext, value_int = agent.get_value(obs[step])
                ext_values[step], int_values[step] = (
                    value_ext.flatten(),
                    value_int.flatten(),
                )
                action, logprob, _, _, _ = agent.get_action_and_value(obs[step])

            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)
            rnd_next_obs = (
                (
                    (next_obs[:, 3, :, :].reshape(args.num_envs, 1, 84, 84) - torch.from_numpy(obs_rms.mean).to(device))
                    / torch.sqrt(torch.from_numpy(obs_rms.var).to(device))
                ).clip(-5, 5)
            ).float()
            target_next_feature = rnd_model.target(rnd_next_obs)
            predict_next_feature = rnd_model.predictor(rnd_next_obs)
            curiosity_rewards[step] = ((target_next_feature - predict_next_feature).pow(2).sum(1) / 2).data
            idx = 0
            for item in info:
                if "episode" in item.keys():
                    avg_returns.append(item['episode']['r'])
                    epi_ret = np.average(avg_returns)
                    print(
                        f"global_step={global_step}, episodic_return={item['episode']['r']}, curiosity_reward={np.mean(curiosity_rewards[step].cpu().numpy())}"
                    )
                    writer.add_scalar("charts/avg_episodic_return", epi_ret, global_step)
                    writer.add_scalar("charts/episodic_return", item['episode']['r'], global_step)
                    writer.add_scalar(
                        "charts/episode_curiosity_reward",
                        curiosity_rewards[step][idx],
                        global_step,
                    )
                    writer.add_scalar("charts/episodic_length", item['episode']['l'], global_step)

                    wandb.log({
                        "charts/avg_episodic_return" : epi_ret,
                        "charts/episodic_return" : item['episode']['r'],
                        "charts/episode_curiosity_reward" :  curiosity_rewards[step][idx],
                        "charts/episodic_length" : item['episode']['l']
                    })
                
                idx += 1

        curiosity_reward_per_env = np.array(
            [discounted_reward.update(reward_per_step) for reward_per_step in curiosity_rewards.cpu().data.numpy().T]
        )
        mean, std, count = (
            np.mean(curiosity_reward_per_env),
            np.std(curiosity_reward_per_env),
            len(curiosity_reward_per_env),
        )
        reward_rms.update_from_moments(mean, std**2, count)

        curiosity_rewards /= np.sqrt(reward_rms.var)

        # bootstrap value if not done
        with torch.no_grad():
            next_value_ext, next_value_int = agent.get_value(next_obs)
            next_value_ext, next_value_int = next_value_ext.reshape(1, -1), next_value_int.reshape(1, -1)
            ext_advantages = torch.zeros_like(rewards, device=device)
            int_advantages = torch.zeros_like(curiosity_rewards, device=device)
            ext_lastgaelam = 0
            int_lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    ext_nextnonterminal = 1.0 - next_done
                    int_nextnonterminal = 1.0
                    ext_nextvalues = next_value_ext
                    int_nextvalues = next_value_int
                else:
                    ext_nextnonterminal = 1.0 - dones[t + 1]
                    int_nextnonterminal = 1.0
                    ext_nextvalues = ext_values[t + 1]
                    int_nextvalues = int_values[t + 1]
                ext_delta = rewards[t] + args.gamma * ext_nextvalues * ext_nextnonterminal - ext_values[t]
                int_delta = curiosity_rewards[t] + args.int_gamma * int_nextvalues * int_nextnonterminal - int_values[t]
                ext_advantages[t] = ext_lastgaelam = (
                    ext_delta + args.gamma * args.gae_lambda * ext_nextnonterminal * ext_lastgaelam
                )
                int_advantages[t] = int_lastgaelam = (
                    int_delta + args.int_gamma * args.gae_lambda * int_nextnonterminal * int_lastgaelam
                )
            ext_returns = ext_advantages + ext_values
            int_returns = int_advantages + int_values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_ext_advantages = ext_advantages.reshape(-1)
        b_int_advantages = int_advantages.reshape(-1)
        b_ext_returns = ext_returns.reshape(-1)
        b_int_returns = int_returns.reshape(-1)
        b_ext_values = ext_values.reshape(-1)

        b_advantages = b_int_advantages * args.int_coef + b_ext_advantages * args.ext_coef

        obs_rms.update(b_obs[:, 3, :, :].reshape(-1, 1, 84, 84).cpu().numpy())

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)

        rnd_next_obs = (
            (
                (b_obs[:, 3, :, :].reshape(-1, 1, 84, 84) - torch.from_numpy(obs_rms.mean).to(device))
                / torch.sqrt(torch.from_numpy(obs_rms.var).to(device))
            ).clip(-5, 5)
        ).float()

        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                predict_next_state_feature, target_next_state_feature = rnd_model(rnd_next_obs[mb_inds])
                forward_loss = F.mse_loss(
                    predict_next_state_feature, target_next_state_feature.detach(), reduction="none"
                ).mean(-1)

                mask = torch.rand(len(forward_loss), device=device)
                mask = (mask < args.update_proportion).type(torch.FloatTensor).to(device)
                forward_loss = (forward_loss * mask).sum() / torch.max(
                    mask.sum(), torch.tensor([1], device=device, dtype=torch.float32)
                )
                _, newlogprob, entropy, new_ext_values, new_int_values = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_ext_values, new_int_values = new_ext_values.view(-1), new_int_values.view(-1)
                if args.clip_vloss:
                    ext_v_loss_unclipped = (new_ext_values - b_ext_returns[mb_inds]) ** 2
                    ext_v_clipped = b_ext_values[mb_inds] + torch.clamp(
                        new_ext_values - b_ext_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    ext_v_loss_clipped = (ext_v_clipped - b_ext_returns[mb_inds]) ** 2
                    ext_v_loss_max = torch.max(ext_v_loss_unclipped, ext_v_loss_clipped)
                    ext_v_loss = 0.5 * ext_v_loss_max.mean()
                else:
                    ext_v_loss = 0.5 * ((new_ext_values - b_ext_returns[mb_inds]) ** 2).mean()

                int_v_loss = 0.5 * ((new_int_values - b_int_returns[mb_inds]) ** 2).mean()
                v_loss = ext_v_loss + int_v_loss
                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + forward_loss

                optimizer.zero_grad()
                loss.backward()
                if args.max_grad_norm:
                    nn.utils.clip_grad_norm_(
                        combined_parameters,
                        args.max_grad_norm,
                    )
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        # Evaluation!
        eval_obs = envs_test.reset()
        eval_done = False
        total_eval_reward = 0

        while not eval_done:
            with torch.no_grad():
                eval_action, _, _, _, _ = agent.get_action_and_value(torch.Tensor(eval_obs).to(device)) 
                eval_obs, eval_reward, eval_done, _ = envs_test.step(eval_action.cpu().numpy())

                rnd_next_eval_obs = (
                (
                    (torch.Tensor(eval_obs[:, 3, :, :].reshape(1, 1, 84, 84)).to(device) - torch.from_numpy(obs_rms.mean).to(device))
                    / torch.sqrt(torch.from_numpy(obs_rms.var).to(device))
                ).clip(-5, 5)
                ).float()
   
                eval_target_next_feature = rnd_model.target(rnd_next_eval_obs)
                eval_predict_next_feature = rnd_model.predictor(rnd_next_eval_obs)
                total_eval_reward += ((eval_target_next_feature - eval_predict_next_feature).pow(2).sum(1) / 2).data

        print("Evaluated the agent and got reward: " + str(total_eval_reward))
        
        if total_eval_reward >= best_reward:
            if not os.path.exists(args.eval_dir):
                os.makedirs(args.eval_dir)
            
            torch.save(agent.state_dict(), args.eval_dir + "/ppo-" + str(update) + ".pt")
            best_reward = total_eval_reward

        wandb.log({
            "eval/reward" : total_eval_reward
        })

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/fwd_loss", forward_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        wandb.log({
            "charts/learning_rate" : optimizer.param_groups[0]["lr"],
            "losses/value_loss" : v_loss.item(),
            "losses/policy_loss" : pg_loss.item(),
            "losses/entropy" : entropy_loss.item(),
            "losses/old_approx_kl" :  old_approx_kl.item(),
            "losses/fwd_loss" : forward_loss.item(),
            "losses/approx_kl" : approx_kl.item(),
            "charts/SPS" : int(global_step / (time.time() - start_time))
        })

    envs.close()
    writer.close()