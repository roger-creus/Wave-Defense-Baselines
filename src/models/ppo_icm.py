import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import WaveDefense

from IPython import embed

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils.utils import *
from utils.networks import *
from utils.icm import *

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
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="WaveDefenseNoReward-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=10000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=16,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.1,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--eval-dir", type=str, default="checkpoints/",
        help="where to save best models")
    
    
    parser.add_argument("--eta", type=float, default=0.2,
        help="the strength of the curiosity reward")
    parser.add_argument("--beta", type=float, default=0.2,
        help="the strength of the inverse model loss compared to forward model loss")
    parser.add_argument("--lambda_", type=float, default=0.1,
        help="the strength of the PPO loss compared to the ICM loss")

    
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
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    envs_test = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(1)]
    )

    agent = Agent(envs).to(device)
    icm = IntrinsicCuriosityModule(in_channels=4, num_actions=envs.single_action_space.n).to(device)

    optimizer = GlobalAdam(list(agent.parameters()) + list(icm.parameters()), lr=args.learning_rate)

    # criterions for ICM: recon loss for fwd model and classification loss for inverse model
    inv_criterion = nn.CrossEntropyLoss()
    fwd_criterion = nn.MSELoss(reduction="none")

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    
    best_reward = 0

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # to store the losses of the ICM
        inv_losses = []
        fwd_losses = []
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            total_reward = 0

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            total_reward = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)


            # one hot actions for labels for ICM inverse model
            action_oh = torch.zeros((args.num_envs, 1, envs.single_action_space.n))
            for e in range(args.num_envs):
                action_oh[e, 0, action[e]] = 1
            action_oh = action_oh.to(device)
            
            # given (s,a,s') fwd model predicts s' from s,a and inverse model predicts a from s,s'
            pred_logits, pred_phi, phi = icm(obs[step], next_obs, action_oh)

            # compute losses
            inv_loss = inv_criterion(pred_logits, action)
            fwd_loss = fwd_criterion(pred_phi, phi) / 2
            
            # derive and add intrinsic reward
            intrinsic_reward = args.eta * torch.mean(fwd_loss.detach(), axis = 1)
            total_reward += intrinsic_reward
            rewards[step] = total_reward

            # save losses
            inv_losses.append(inv_loss)
            fwd_losses.append(torch.mean(fwd_loss))

            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                    
                    wandb.log({
                        "charts/episodic_return" : item["episode"]["r"],
                        "charts/episodic_length" : item["episode"]["l"]
                    }, step = global_step)
                    break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # total inv and fwd loss of the ICM model in this iteration of rollouts 
        inv_loss = torch.sum(torch.tensor(inv_losses)).to(device)
        fwd_loss = torch.sum(torch.tensor(fwd_losses)).to(device)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
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
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = args.lambda_ * (pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef)

                # compute curiosity loss
                curiosity_loss = (1 - args.beta) * inv_loss + args.beta * fwd_loss

                loss += curiosity_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
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
                eval_action, _, _, _ = agent.get_action_and_value(torch.Tensor(eval_obs).to(device)) 
                eval_obs_new, eval_reward, eval_done, _ = envs_test.step(eval_action.cpu().numpy())
                
                # one hot actions for labels for ICM inverse model
                eval_action_oh = torch.zeros((1, 1, envs_test.single_action_space.n))
                eval_action_oh[0, 0, eval_action[0]] = 1
                eval_action_oh = eval_action_oh.to(device)
                
                # given (s,a,s') fwd model predicts s' from s,a and inverse model predicts a from s,s'
                eval_pred_logits, eval_pred_phi, eval_phi = icm(torch.Tensor(eval_obs).to(device), torch.Tensor(eval_obs_new).to(device), eval_action_oh)

                # compute losses
                eval_fwd_loss = fwd_criterion(eval_pred_phi, eval_phi) / 2
                
                # derive and add intrinsic reward
                eval_intrinsic_reward = args.eta * torch.mean(eval_fwd_loss.detach(), axis = 1)
                total_eval_reward += eval_intrinsic_reward.item()

                eval_obs = eval_obs_new

        print("Evaluated the agent and got reward: " + str(total_eval_reward))
        
        if total_eval_reward >= best_reward:
            if not os.path.exists(args.eval_dir):
                os.makedirs(args.eval_dir)
            
            torch.save(agent.state_dict(), args.eval_dir + "/ppo-" + str(update) + ".pt")
            best_reward = total_eval_reward

        wandb.log({
            "eval/reward" : total_eval_reward
        })

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        
        wandb.log({
            "charts/learning_rate" : optimizer.param_groups[0]["lr"],
            "losses/value_loss" :  v_loss.item(),
            "losses/policy_loss" :  pg_loss.item(),
            "losses/entropy" : entropy_loss.item(),
            "losses/old_approx_kl" :  old_approx_kl.item(),
            "losses/approx_kl" : approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "losses/explained_variance" : explained_var,
            "icm/fwd_loss" : fwd_loss.item(),
            "icm/inv_loss" : inv.item(),
            "icm/curiosity_loss" : curiosity_loss.item(),
            "charts/SPS" : int(global_step / (time.time() - start_time))
        }, step = global_step)

        # clean cuda objects
        torch.cuda.empty_cache()

    envs.close()
    writer.close()