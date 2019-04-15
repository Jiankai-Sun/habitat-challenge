#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


EPS_PPO = 1e-5


class PPO(nn.Module):
    def __init__(
        self,
        actor_critic,
        icm_model,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        value_loss_coef,
        entropy_coef,
        beta,
        prediction_lr_scale,
        device=None,
        lr=None,
        eps=None,
        max_grad_norm=None,
        use_clipped_value_loss=True,
        eta=0.01
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

        self.icm_model = icm_model

        self.device = device
        self.eta = eta
        self.beta = beta
        self.prediction_lr_scale = prediction_lr_scale
        if self.icm_model is None:
            self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        else:
            self.optimizer = optim.Adam(list(actor_critic.parameters()) + list(icm_model.parameters()), lr=lr, eps=eps)
            self.ce = nn.CrossEntropyLoss()
            self.forward_mse = nn.MSELoss()

            # self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

    def forward(self, *x):
        raise NotImplementedError

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + EPS_PPO
        )

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        inverse_loss_epoch = 0
        forward_loss_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for sample in data_generator:
                (
                    obs_batch,
                    next_obs_batch,
                    recurrent_hidden_states_batch,
                    actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                ) = sample

                # --------------------------------------------------------------------------------
                # for Curiosity-driven
                if self.icm_model is not None:
                    # print('actions_batch.shape[0]: ', actions_batch.shape[0]) # 128
                    action_onehot = torch.FloatTensor(actions_batch.shape[0], self.icm_model.output_size).to(self.device)
                    action_onehot.zero_()
                    action_onehot.scatter_(1, actions_batch.view(-1, 1), 1)
                    real_next_state_feature, pred_next_state_feature, pred_action = self.icm_model(
                        [obs_batch['rgb'].permute(0, 3, 1, 2).to(self.device), next_obs_batch['rgb'].permute(0, 3, 1, 2).to(self.device), action_onehot])

                    inverse_loss = (1 - self.beta) * self.ce(pred_action, actions_batch.squeeze())

                    forward_loss = self.beta * self.forward_mse(
                        pred_next_state_feature, real_next_state_feature.detach())
                # ---------------------------------------------------------------------------------

                # Reshape to do in a single forward pass for all steps
                (
                    values,
                    action_log_probs,
                    dist_entropy,
                    _,
                ) = self.actor_critic.evaluate_actions(
                    obs_batch,
                    recurrent_hidden_states_batch,
                    masks_batch,
                    actions_batch,
                )

                ratio = torch.exp(
                    action_log_probs - old_action_log_probs_batch
                )
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    * adv_targ
                )
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch
                    ).pow(2)
                    value_loss = (
                        0.5
                        * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                if self.icm_model is None:
                    (
                        value_loss * self.value_loss_coef
                        + action_loss
                        - dist_entropy * self.entropy_coef
                    ).backward()
                else:
                    # print(value_loss * self.value_loss_coef, action_loss, dist_entropy * self.entropy_coef, forward_loss, inverse_loss, (inverse_loss + forward_loss) * self.prediction_lr_scale)
                    (
                        value_loss * self.value_loss_coef
                        + action_loss
                        - dist_entropy * self.entropy_coef
                        + (inverse_loss + forward_loss) * self.prediction_lr_scale
                    ).backward()
                nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                inverse_loss_epoch += inverse_loss.item()
                forward_loss_epoch += forward_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        inverse_loss_epoch /= num_updates
        forward_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, inverse_loss_epoch, forward_loss_epoch

    def compute_intrinsic_reward(self, state, next_state, action):
        '''

        :param state:
        :param next_state:
        :param action: shape: (2,1)
        :return:
        '''
        # state = state.permute(0, 3, 1, 2)
        # next_state = next_state.permute(0, 3, 1, 2)

        state = torch.FloatTensor(state).permute(0, 3, 1, 2).to(self.device)
        next_state = torch.FloatTensor(next_state).permute(0, 3, 1, 2).to(self.device)
        action = torch.LongTensor(action).to(self.device)

        action_onehot = torch.FloatTensor(
            len(action), self.icm_model.output_size).to(device=self.device)
        action_onehot.zero_()
        action_onehot.scatter_(1, action.view(len(action), -1), 1)

        real_next_state_feature, pred_next_state_feature, pred_action = self.icm_model(
            [state, next_state, action_onehot])
        intrinsic_reward = self.eta * F.mse_loss(real_next_state_feature, pred_next_state_feature, reduction='none').mean(-1, keepdim=True)
        return intrinsic_reward.data.cpu().numpy()

