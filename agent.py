import gymnasium as gym
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
from torch import nn
import yaml

from replay_memory import ReplayMemory
from dqn import DQN

from datetime import datetime, timedelta
import argparse
import itertools

import flappy_bird_gymnasium
import os

DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

RUNS_DIR = 'runs'

os.makedirs(RUNS_DIR, exist_ok=True)

matplotlib.use('Agg')


class Agent:
    def __init__(self, hyperparam_option):
        with open('hyperparameters.yml', 'r') as f:
            all_hyperparams = yaml.safe_load(f)
            self.hyperparams = all_hyperparams[hyperparam_option]
            self.hyperparam_option = hyperparam_option
            self.env_id = self.hyperparams['env_id']
            self.replay_memory_size = self.hyperparams['replay_memory_size']  # size of the replay memory
            self.mini_batch_size = self.hyperparams[
                'mini_batch_size']  # size of the training data set sampled from the replay memory
            self.epsilon_init = self.hyperparams['epsilon_init']  # proportion of actions that are random
            self.epsilon_decay = self.hyperparams['epsilon_decay']  # decay rate of epsilon
            self.epsilon_min = self.hyperparams['epsilon_min']  # minimum value of epsilon
            self.network_sync_rate = self.hyperparams['network_sync_rate']  # how often to update the target network
            self.learning_rate_a = self.hyperparams['learning_rate_a']
            self.discount_factor_g = self.hyperparams[
                'discount_factor_g']  # how much to discount future rewardsvs sooner rewards
            self.stop_on_reward = self.hyperparams['stop_on_reward']  # stop training when the reward reaches this value
            self.fc1_nodes = self.hyperparams['fc1_nodes']
            self.env_make_params = self.hyperparams.get('env_make_params', {})
            self.enable_double_dqn = self.hyperparams.get('enable_double_dqn', False)
            self.use_cuda = self.hyperparams.get('use_cuda', torch.cuda.is_available())
            self.device = torch.device('cuda' if self.use_cuda else 'cpu')
            self.max_iter = self.hyperparams.get('max_iter', 1000000)
            self.rewards_to_average = self.hyperparams.get('rewards_to_average', 1)

            self.loss_fn = nn.MSELoss()  # loss function (mean squared error)
            self.optimizer = None

            self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparam_option}.log')
            self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparam_option}.pt')
            self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparam_option}.png')

    def run(self, is_train, render=False):
        if is_train:
            start_time = datetime.now()
            last_graph_update_time = start_time

            log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
            print(log_message)
            with open(self.LOG_FILE, 'w') as file:
                file.write(log_message + '\n')
        # run the agent
        env = gym.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)

        num_actions = env.action_space.n  # number of output options
        num_states = env.observation_space.shape[0]  # number of input nodes
        rewards_per_episode = []  # track rewards

        policy_net = DQN(num_states, num_actions, self.fc1_nodes).to(self.device)  # the policy network

        if is_train:
            memory = ReplayMemory(capacity=10000)
            epsilon = self.epsilon_init
            target_net = DQN(num_states, num_actions, self.fc1_nodes).to(self.device)
            target_net.load_state_dict(policy_net.state_dict())

            # List to keep track of epsilon decay
            epsilon_history = []

            step_count = 0

            # policy network optimizer
            self.optimizer = torch.optim.Adam(policy_net.parameters(), lr=self.learning_rate_a)

            best_reward = -float('inf')
        else:
            print(f"Loading model from {self.MODEL_FILE}")
            policy_net.load_state_dict(torch.load(self.MODEL_FILE))
            policy_net.eval()

        for episode in range(self.max_iter):
            state, _ = env.reset()
            # convert anything going into the network to a tensor
            state = torch.tensor(state, dtype=torch.float, device=self.device).to(self.device)

            episode_reward = 0.0
            done = False
            while not done and episode_reward < self.stop_on_reward:
                # Picking an action
                if is_train and random.random() < epsilon:
                    # random action
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=self.device)
                else:
                    with torch.no_grad():
                        # action with best Q value
                        action = policy_net(state.unsqueeze(dim=0)).squeeze().argmax()

                # Processing
                new_state, reward, done, truncated, info = env.step(action.item())

                # accumulate reward
                episode_reward += reward

                # convert new state and reward to tensors on device
                new_state = torch.tensor(new_state, dtype=torch.float, device=self.device)
                reward = torch.tensor(reward, dtype=torch.float, device=self.device)

                if is_train:
                    memory.append((state, action, new_state, reward, done))

                    step_count += 1

                state = new_state

            rewards_per_episode.append(episode_reward)

            # Save model when new best reward is obtained.
            if is_train:
                last_n_reward_avg = np.mean(rewards_per_episode[-self.rewards_to_average:])
                if last_n_reward_avg > best_reward:
                    log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best avg reward {last_n_reward_avg:0.1f} ({(last_n_reward_avg - best_reward) / best_reward * 100:+.1f}%) at episode {episode}, saving model..."
                    print(log_message)
                    with open(self.LOG_FILE, 'a') as file:
                        file.write(log_message + '\n')

                    torch.save(policy_net.state_dict(), self.MODEL_FILE)
                    best_reward = last_n_reward_avg

                # Update graph every x seconds
                current_time = datetime.now()
                if current_time - last_graph_update_time > timedelta(seconds=10):
                    self.save_graph(rewards_per_episode, epsilon_history)
                    last_graph_update_time = current_time

                # If enough experience has been collected
                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_net, target_net)

                    # Decay epsilon
                    # in this implementation we're using a geometric decay for epsilon (taking the product of epsilon_decay and current epsilon)
                    # a linear decay is another option, decreasing epsilon by a fixed amount each episode (adjust epsilon_decay hyperparameter accordingly)
                    epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
                    epsilon_history.append(epsilon)

                    # Copy policy network to target network after a certain number of steps
                    if step_count > self.network_sync_rate:
                        target_net.load_state_dict(policy_net.state_dict())
                        step_count = 0

    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x - 99):(x + 1)])
        plt.subplot(121)  # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122)  # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.GRAPH_FILE)
        plt.close(fig)

    def optimize(self, mini_batch, policy_net, target_net):
        # slow but easy to understand version
        # for state, action, new_state, reward, done in mini_batch:
        #     if done:
        #         target_q = reward
        #     else:
        #     # calculate the target value
        #         with torch.no_grad():
        #             target_q = reward + self.discount_factor_g * target_net(new_state).max()

        #     current_q = policy_net(state)

        #     loss = self.loss_fn(current_q, target_q)

        #     self.optimizer.zero_grad() # clear the gradients
        #     loss.backward() # compute gradients (backpropagation)
        #     self.optimizer.step()

        # fast version
        # transpose the batch of experiences
        states, actions, new_states, rewards, dones = zip(*mini_batch)

        # stack tensors to create batch tensors
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.stack(rewards)
        dones = torch.tensor(dones).float().to(self.device)

        with torch.no_grad():
            if self.enable_double_dqn:
                best_action_from_policy = policy_net(new_states).argmax(dim=1)
                target_q = rewards + (1 - dones) * self.discount_factor_g * \
                           target_net(new_states).gather(dim=1,
                                                         index=best_action_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                # calculate target q values (expected future rewards)
                target_q = rewards + (1 - dones) * self.discount_factor_g * target_net(new_states).max(dim=1)[0]

        # calculate the Q value from the current policy
        current_q = policy_net(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()  # clear the gradients
        loss.backward()  # compute gradients (backpropagation)
        self.optimizer.step()


if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('model', help='')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparam_option=args.model)

    if args.train:
        dql.run(is_train=True)
    else:
        dql.run(is_train=False, render=True)
