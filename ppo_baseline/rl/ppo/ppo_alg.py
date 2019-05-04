#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim


EPS_PPO = 1e-5


class PPO(nn.Module):
    def __init__(
        self,
        actor_critic,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
        use_clipped_value_loss=True,
    ):

        super().__init__()

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, *x):
        raise NotImplementedError

    def update(self, rollouts):
        action_loss_epoch = 0

        data_generator = rollouts.recurrent_generator(
            self.num_mini_batch
        )

        for sample in data_generator:
            (
                obs_batch,
                recurrent_hidden_states_batch,
                actions_batch,
                value_preds_batch,
                masks_batch,
                st_actions_batch,
            ) = sample

            print(value_preds_batch[0], st_actions_batch.squeeze(1)[0])
            action_loss = self.criterion(value_preds_batch, st_actions_batch.squeeze(1))

            self.optimizer.zero_grad()
            action_loss.backward()
            nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), self.max_grad_norm
            )
            self.optimizer.step()

            action_loss_epoch += action_loss.item()

        num_updates = self.ppo_epoch * rollouts.step

        action_loss_epoch /= num_updates

        return action_loss_epoch
