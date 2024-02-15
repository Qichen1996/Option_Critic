import os
from argparse import ArgumentParser
import sys

import numpy as np
import torch
from copy import deepcopy
from gymnasium.spaces import Discrete
from utils import sys, time, trange, notice, pd, kwds_str

from .option_critic import OptionCriticFeatures
from .option_critic import critic_loss as critic_loss_fn
from .option_critic import actor_loss as actor_loss_fn

from .experience_replay import ReplayBuffer

from .trainer import BaseTrainer
from .logger import Logger


def get_dqn_config():
    parser = ArgumentParser()

    parser.add_argument("--buffer-size", type=int, default=20000,
        help="the replay memory buffer size")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.03,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=20000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")

    parser.add_argument('--env', default='CartPole-v0', help='ROM to run')
    parser.add_argument('--optimal-eps', type=float, default=0.05, help='Epsilon when playing optimally')
    parser.add_argument('--frame-skip', default=4, type=int, help='Every how many frames to process')
    parser.add_argument('--learning-rate', type=float, default=.0005, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=.99, help='Discount rate')
    parser.add_argument('--epsilon-start', type=float, default=1.0, help=('Starting value for epsilon.'))
    parser.add_argument('--epsilon-min', type=float, default=.1, help='Minimum epsilon.')
    parser.add_argument('--epsilon-decay', type=float, default=20000, help=('Number of steps to minimum epsilon.'))
    parser.add_argument('--max-history', type=int, default=10000, help=('Maximum number of steps stored in replay'))
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size.')
    parser.add_argument('--freeze-interval', type=int, default=200, help=('Interval between target freezes.'))
    parser.add_argument('--update-frequency', type=int, default=4, help=('Number of actions before each SGD update.'))
    parser.add_argument('--termination-reg', type=float, default=0.01,
                        help=('Regularization to decrease termination prob.'))
    parser.add_argument('--entropy-reg', type=float, default=0.01, help=('Regularization to increase policy entropy.'))
    parser.add_argument('--num-options', type=int, default=3, help=('Number of options to create.'))
    parser.add_argument('--temp', type=float, default=1, help='Action distribution softmax tempurature param.')

    parser.add_argument('--max_steps_ep', type=int, default=18000, help='number of maximum steps per episode.')
    parser.add_argument('--max_steps_total', type=int, default=int(4e6),
                        help='number of maximum steps to take.')  # bout 4 million
    parser.add_argument('--cuda', type=bool, default=True, help='Enable CUDA training (recommended if possible).')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for numpy, torch, random.')
    parser.add_argument('--logdir', type=str, default='runs', help='Directory for logging statistics')
    parser.add_argument('--exp', type=str, default=None, help='optional experiment name')
    parser.add_argument('--switch-goal', type=bool, default=False, help='switch goal after 2k eps')
    
    return parser


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def to_tensor(obs):
    obs = np.asarray(obs)
    obs = torch.from_numpy(obs).float()
    return obs

class DQNTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

        self.cent_observation_space = self.envs.cent_observation_space
        self.observation_space = self.envs.observation_space[0]
        self.action_space = self.envs.action_space[0]


    def train(self):
        env = self.envs
        args = self.all_args
        option_critic = OptionCriticFeatures
        device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

        option_critic = option_critic(
            in_features=self.cent_observation_space.shape[0],
            action_space=Discrete(4),
            num_options=args.num_options,
            temperature=args.temp,
            eps_start=args.epsilon_start,
            eps_min=args.epsilon_min,
            eps_decay=args.epsilon_decay,
            eps_test=args.optimal_eps,
            device=device
        )
        # Create a prime network for more stable Q values
        option_critic_prime = deepcopy(option_critic)

        optim = torch.optim.RMSprop(option_critic.parameters(), lr=args.learning_rate)

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        buffer = ReplayBuffer(capacity=args.max_history, seed=args.seed)
        logger = Logger(logdir=args.logdir, run_name=f"{OptionCriticFeatures.__name__}-{args.env}-{args.exp}")

        steps = 0;

        while steps < self.num_env_steps:

            rewards = 0;
            option_lengths = {opt: [] for opt in range(args.num_options)}

            _, obs, _ = env.reset()
            obs = obs.reshape(-1)
            state = option_critic.get_state(to_tensor(obs))
            greedy_option = option_critic.greedy_option(state)
            current_option = 0

            # Goal switching experiment: run for 1k episodes in fourrooms, switch goals and run for another
            # 2k episodes. In option-critic, if the options have some meaning, only the policy-over-options
            # should be finedtuned (this is what we would hope).
            # if args.switch_goal and logger.n_eps == 1000:
            #     torch.save({'model_params': option_critic.state_dict(),
            #                 'goal_state': env.goal},
            #                f'models/option_critic_seed={args.seed}_1k')
            #     env.switch_goal()
            #     print(f"New goal {env.goal}")
            #
            # if args.switch_goal and logger.n_eps > 2000:
            #     torch.save({'model_params': option_critic.state_dict(),
            #                 'goal_state': env.goal},
            #                f'models/option_critic_seed={args.seed}_2k')
            #     break

            done = False;
            option_termination = True;
            ep_steps = 0
            curr_op_len = 0
            episodes = 0
            while not done:
                epsilon = option_critic.epsilon

                if option_termination:
                    option_lengths[current_option].append(curr_op_len)
                    current_option = np.random.choice(args.num_options) if np.random.rand() < epsilon else greedy_option
                    curr_op_len = 0

                action, logp, entropy = option_critic.get_action(state, current_option)

                actions = [[[current_option, action]]]
                _, next_obs, reward, done, infos, _ = env.step(actions)

                next_obs = next_obs.reshape(-1)
                reward = reward[0][0][0]
                done = done[0]
                buffer.push(obs, current_option, reward, next_obs, done)
                rewards += reward

                actor_loss, critic_loss = None, None
                if len(buffer) > args.batch_size:
                    actor_loss = actor_loss_fn(obs, current_option, logp, entropy, \
                                               reward, done, next_obs, option_critic, option_critic_prime, args)
                    loss = actor_loss

                    if steps % args.update_frequency == 0:
                        data_batch = buffer.sample(args.batch_size)
                        critic_loss = critic_loss_fn(option_critic, option_critic_prime, data_batch, args)
                        loss += critic_loss

                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    if steps % args.freeze_interval == 0:
                        option_critic_prime.load_state_dict(option_critic.state_dict())

                state = option_critic.get_state(to_tensor(next_obs))
                option_termination, greedy_option = option_critic.predict_option_termination(state, current_option)
                logger.log_data(steps, actor_loss, critic_loss, entropy.item(), epsilon)

                # update global steps etc
                steps += 1
                curr_op_len += 1
                ep_steps += 1
                obs = next_obs

            episodes += 1
            if episodes % self.log_interval == 0:
                rew_df = pd.concat([pd.DataFrame(d['step_rewards']) for d in infos])
                rew_info = rew_df.describe().loc[['mean', 'std', 'min', 'max']].unstack()
                rew_info.index = ['_'.join(idx) for idx in rew_info.index]
                self.log_train(rew_info, steps)
                logger.log_episode(steps, rewards, option_lengths, ep_steps, epsilon)

    def take_actions(self, obs):
        return np.array([self.q_net(torch.Tensor(ob).to(self.device))
                         .cpu().numpy() for ob in obs])

    def save(self, version=''):
        path = os.path.join(self.save_dir, f"dqn{version}.pt")
        notice(f"Saving model to {path}")
        torch.save(self.q_net.state_dict(), path)

    def load(self, version=''):
        path = os.path.join(self.model_dir, f"dqn{version}.pt")
        notice(f"Loading model from {path}")
        self.q_net.load_state_dict(torch.load(path))
        self.targ_net.load_state_dict(self.q_net.state_dict())
