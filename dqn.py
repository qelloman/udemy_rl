
import numpy as np
import random
import copy
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset

from torch.optim import AdamW
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
import gym
from gym.wrappers import TimeLimit, RecordEpisodeStatistics

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_gpu = torch.cuda.device_count()

class DQN(nn.Module):
    def __init__(self, hidden_unit, n_obs, n_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, hidden_unit),
            nn.ReLU(),
            nn.Linear(hidden_unit, hidden_unit),
            nn.ReLU(),
            nn.Linear(hidden_unit, n_action)
        )
    
    def forward(self, x):
        out = self.net(x.float())
        return out

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def append(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


class RLDataset(IterableDataset):
    def __init__(self, buffer, sample_per_epoch):
        self.buffer = buffer
        self.sample_per_epoch = sample_per_epoch

    def __iter__(self):
        for exp in self.buffer.sample(self.sample_per_epoch):
            yield exp

def create_env(env_name):
    env = gym.make(env_name)
    env = TimeLimit(env, max_episode_steps=400)
    env = RecordEpisodeStatistics(env)
    return env

def epsilon_greedy(env, net, state, eps=0.0):
    if np.random.random() < eps:
        action = env.action_space.sample()
    else:
        state = torch.tensor([state]).to(device)
        q_values = net(state)
        _, action = q_values.max(1, keepdim=True)
        action = int(action.item())
    return action

class DeepQLearning(LightningModule):
    def __init__(self, env_name, hidden_unit=128, capacity=100_000,
                policy=epsilon_greedy, eps_start=1.0, eps_end=0.15, gamma=0.99,
                eps_last_period=100, optim=AdamW, lr=1e-3, batch_size=128,
                sample_per_epoch=10_000, sync_period=10, loss_fn=F.smooth_l1_loss):
        super().__init__()
        self.env = create_env(env_name)
        n_obs = self.env.observation_space.shape[0]
        n_action = self.env.action_space.n

        self.q_net = DQN(hidden_unit, n_obs, n_action)
        self.target_q_net = copy.deepcopy(self.q_net)

        self.buffer = ReplayBuffer(capacity)

        self.save_hyperparameters()

        while len(self.buffer) < self.hparams.sample_per_epoch:
            print("filling")
            self.play_episode(eps=0) 
    
    @torch.no_grad()
    def play_episode(self, policy=None, eps=0.0):
        state = self.env.reset()
        done = False

        while not done:
            if policy:
                action = policy(self.env, self.q_net, state, eps)
            else:
                action = self.env.action_space.sample()
            next_state, reward, done, info = self.env.step(action)
            exp = (state, action, reward, done, next_state)
            self.buffer.append(exp)
            state = next_state

    def forward(self, x):
        return self.q_net(x)

    def configure_optimizers(self):
        optim = self.hparams.optim(self.q_net.parameters(), self.hparams.lr)
        return [optim]

    def train_dataloader(self):
        dataset = RLDataset(self.buffer, self.hparams.sample_per_epoch)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size
        )
        return dataloader

    def training_step(self, batch, batch_idx):
        states, actions, rewards, dones, next_states = batch
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        action_values = self.q_net(states).gather(1, actions)
        next_q_values, _ = self.target_q_net(next_states).max(1, keepdim=True)
        next_q_values[dones] = 0.0

        expected_action_values = rewards + self.hparams.gamma * next_q_values

        loss = self.hparams.loss_fn(expected_action_values, action_values)
        self.log("episode/loss", loss)
        return loss

    def training_epoch_end(self, training_outputs):
        eps = max(
            self.hparams.eps_end,
            self.hparams.eps_start - self.current_epoch / self.hparams.eps_last_period
        )
        self.play_episode(self.hparams.policy, eps)
        self.log("episode/return", self.env.return_queue[-1])

        if self.current_epoch % self.hparams.sync_period == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

algo = DeepQLearning("LunarLander-v2")

trainer = Trainer(
    gpus=num_gpu,
    max_epochs=100_000,
    callbacks=[EarlyStopping(monitor="episode/return", patience=500, mode='max')]
)
trainer.fit(algo)