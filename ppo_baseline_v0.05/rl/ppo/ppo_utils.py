#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CustomFixedCategorical(torch.distributions.Categorical):
    def sample(self, sample_shape=torch.Size()):
        return super().sample(sample_shape).unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
                .log_prob(actions.squeeze(-1))
                .view(actions.size(0), -1)
                .sum(-1)
                .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class CategoricalNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.linear = nn.Linear(num_inputs, num_outputs)

        nn.init.orthogonal_(self.linear.weight, gain=0.01)
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, x):
        x = self.linear(x)
        return CustomFixedCategorical(logits=x)


def _flatten_helper(t, n, tensor):
    return tensor.view(t * n, *tensor.size()[2:])


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class RolloutStorage:
    def __init__(
            self,
            num_steps,
            num_envs,
            observation_space,
            action_space,
            recurrent_hidden_state_size,
    ):
        self.observations = {}
        self.observation_space = observation_space
        for sensor in observation_space.spaces:
            self.observations[sensor] = torch.zeros(
                num_steps + 1,
                num_envs,
                *observation_space.spaces[sensor].shape
            )

        self.recurrent_hidden_states = torch.zeros(
            num_steps + 1, num_envs, recurrent_hidden_state_size
        )

        self.rewards = torch.zeros(num_steps, num_envs, 1)
        self.value_preds = torch.zeros(num_steps + 1, num_envs, 4)
        if action_space.__class__.__name__ == "Discrete":
            action_shape = 1
        else:
            action_shape = action_space.shape[0]

        self.actions = torch.zeros(num_steps, num_envs, action_shape)
        self.st_actions = torch.zeros(num_steps, num_envs, action_shape)
        if action_space.__class__.__name__ == "Discrete":
            self.actions = self.actions.long()
            self.st_actions = self.st_actions.long()

        self.masks = torch.ones(num_steps + 1, num_envs, 1)

        self.num_steps = num_steps
        self.num_envs = num_envs
        self.recurrent_hidden_state_size = recurrent_hidden_state_size
        self.step = 0

    def to(self, device):
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)

        self.recurrent_hidden_states = self.recurrent_hidden_states.to(device)
        self.value_preds = self.value_preds.to(device)
        self.actions = self.actions.to(device)
        self.st_actions = self.st_actions.to(device)
        self.masks = self.masks.to(device)

    def insert(
            self,
            observations,
            recurrent_hidden_states,
            actions,
            value_preds,
            masks,
            st_actions,
    ):
        for sensor in observations:
            self.observations[sensor][self.step + 1] = observations[sensor]
        self.recurrent_hidden_states[self.step + 1] = recurrent_hidden_states
        self.actions[self.step] = actions
        self.st_actions[self.step] = st_actions
        self.value_preds[self.step] = value_preds
        self.masks[self.step + 1] = masks

        self.step = (self.step + 1) % self.num_steps

    def after_update(self, done):
        # for sensor in self.observations:
        #     self.observations[sensor][0]=self.observations[sensor][-1])
        # for sensor in self.observations:
        #     torch.zeros(
        #         self.num_steps + 1,
        #         self.num_envs,
        #         *self.observation_space.spaces[sensor].shape)
        #
        # self.recurrent_hidden_states = torch.zeros(
        #     self.num_steps + 1, self.num_envs, self.recurrent_hidden_state_size
        # )
        # self.masks = torch.ones(self.num_steps + 1, self.num_envs, 1)
        # self.step = 0
        if done:
            for sensor in self.observations:
                torch.zeros(
                    self.num_steps + 1,
                    self.num_envs,
                    *self.observation_space.spaces[sensor].shape)

            self.recurrent_hidden_states = torch.zeros(
                self.num_steps + 1, self.num_envs, self.recurrent_hidden_state_size
            )
            self.masks = torch.ones(self.num_steps + 1, self.num_envs, 1)
            self.step = 0
        else:
            for sensor in self.observations:
                self.observations[sensor][0] = self.observations[sensor][-1]

            self.recurrent_hidden_states[0] = self.recurrent_hidden_states[-1]
            self.masks[0] = self.masks[-1]
            self.step = 0
            # self.observations = self.observations.detach()
            # self.recurrent_hidden_states = self.recurrent_hidden_states.detach()
            # self.masks = self.masks.detach()

    def recurrent_generator(self, num_mini_batch):
        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch = defaultdict(list)

            recurrent_hidden_states_batch = []
            actions_batch = []
            st_actions_batch = []
            value_preds_batch = []
            masks_batch = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]

                for sensor in self.observations:
                    observations_batch[sensor].append(
                        self.observations[sensor][:-1, ind]
                    )

                recurrent_hidden_states_batch.append(
                    self.recurrent_hidden_states[0:1, ind]
                )

                actions_batch.append(self.actions[:, ind])
                st_actions_batch.append(self.st_actions[:, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])

            T, N = self.num_steps, num_envs_per_batch

            # These are all tensors of size (T, N, -1)
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(
                    observations_batch[sensor], 1
                )

            actions_batch = torch.stack(actions_batch, 1)
            st_actions_batch = torch.stack(st_actions_batch, 1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)

            # States is just a (N, -1) tensor
            recurrent_hidden_states_batch = torch.stack(
                recurrent_hidden_states_batch, 1
            ).view(N, -1)

            # Flatten the (T, N, ...) tensors to (T * N, ...)
            for sensor in observations_batch:
                observations_batch[sensor] = _flatten_helper(
                    T, N, observations_batch[sensor]
                )

            actions_batch = _flatten_helper(T, N, actions_batch)
            st_actions_batch = _flatten_helper(T, N, st_actions_batch)
            value_preds_batch = _flatten_helper(T, N, value_preds_batch)
            masks_batch = _flatten_helper(T, N, masks_batch)

            yield (
                observations_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                value_preds_batch,
                masks_batch,
                st_actions_batch,
            )


def batch_obs(observations, device):
    batch = defaultdict(list)

    for obs in observations:
        for sensor in obs:
            batch[sensor].append(obs[sensor])

    for sensor in batch:
        batch[sensor] = torch.tensor(
            np.array(batch[sensor]), dtype=torch.float
        ).to(device)
    return batch


def ppo_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip-param",
        type=float,
        default=0.2,
        help="ppo clip parameter (default: 0.2)",
    )
    parser.add_argument(
        "--ppo-epoch",
        type=int,
        default=4,
        help="number of ppo epochs (default: 4)",
    )
    parser.add_argument(
        "--num-mini-batch",
        type=int,
        default=32,
        help="number of batches for ppo (default: 32)",
    )
    parser.add_argument(
        "--value-loss-coef",
        type=float,
        default=0.5,
        help="value loss coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="entropy term coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--lr", type=float, default=7e-4, help="learning rate (default: 7e-4)"
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-5,
        help="RMSprop optimizer epsilon (default: 1e-5)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="max norm of gradients (default: 0.5)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=5,
        help="number of forward steps in A2C (default: 5)",
    )
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument(
        "--num-processes",
        type=int,
        default=16,
        help="number of training processes " "to use (default: 16)",
    )
    parser.add_argument(
        "--use-gae",
        action="store_true",
        default=False,
        help="use generalized advantage estimation",
    )
    parser.add_argument(
        "--use-linear-lr-decay",
        action="store_true",
        default=False,
        help="use a linear schedule on the learning rate",
    )
    parser.add_argument(
        "--use-linear-clip-decay",
        action="store_true",
        default=False,
        help="use a linear schedule on the " "ppo clipping parameter",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="discount factor for rewards (default: 0.99)",
    )
    parser.add_argument(
        "--tau", type=float, default=0.95, help="gae parameter (default: 0.95)"
    )
    parser.add_argument(
        "--log-file", type=str, required=True, help="path for log file"
    )
    parser.add_argument(
        "--tb-log-dir", type=str, required=True, help="path for Tensorboard log directory"
    )
    parser.add_argument(
        "--reward-window-size",
        type=int,
        default=50,
        help="logging window for rewards",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        help="number of updates after which metrics are logged",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50,
        help="number of updates after which models are checkpointed",
    )
    parser.add_argument(
        "--checkpoint-folder",
        type=str,
        required=True,
        help="folder for storing checkpoints",
    )
    parser.add_argument(
        "--sim-gpu-id",
        nargs='+',
        type=int,
        required=True,
        default=[],
        help="gpu id on which scenes are loaded",
    )
    parser.add_argument(
        "--pth-gpu-id",
        type=int,
        required=True,
        help="gpu id on which pytorch runs",
    )
    parser.add_argument(
        "--num-updates",
        type=int,
        default=10000,
        help="number of PPO updates to run",
    )
    # parser.add_argument(
    #     "--sensors",
    #     type=str,
    #     default="RGB_SENSOR,DEPTH_SENSOR",
    #     help="comma separated string containing different sensors to use,"
    #     "currently 'RGB_SENSOR' and 'DEPTH_SENSOR' are supported",
    # )
    parser.add_argument(
        "--task-config",
        type=str,
        default="tasks/pointnav.yaml",
        help="path to config yaml containing information about task",
    )
    parser.add_argument("--seed", type=int, default=100)

    return parser
