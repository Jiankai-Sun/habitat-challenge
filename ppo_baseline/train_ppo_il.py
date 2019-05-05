#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from time import time
from collections import deque
import random
import numpy as np

import torch
import torch.nn as nn
import habitat
from habitat import logger
# challenge API
# from habitat.sims.habitat_simulator import SimulatorActions, SIM_NAME_TO_ACTION
from habitat.sims.habitat_simulator import SimulatorActions
from habitat.config.default import get_config as cfg_env
from config.default import cfg as cfg_baseline
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
# from rl.ppo import PPO, Policy,
from rl.ppo.ppo_alg import PPO
from rl.ppo.policy import Policy
from rl.ppo.ppo_utils import update_linear_schedule, ppo_args, batch_obs, RolloutStorage
from tensorboardX import SummaryWriter
import torch.optim as optim

from shortest_path_follower import ShortestPathFollower, name2action

class NavRLEnv(habitat.RLEnv):
    def __init__(self, config_env, config_baseline, dataset):
        self._config_env = config_env.TASK
        self._config_baseline = config_baseline
        self._previous_target_distance = None
        self._previous_action = None
        self._episode_distance_covered = None
        super().__init__(config_env, dataset)

    def reset(self):
        self._previous_action = None

        observations = super().reset()

        self._previous_target_distance = self.habitat_env.current_episode.info[
            "geodesic_distance"
        ]
        return observations

    def step(self, action):
        self._previous_action = action
        return super().step(action)

    def get_reward_range(self):
        return (
            self._config_baseline.BASELINE.RL.SLACK_REWARD - 1.0,
            self._config_baseline.BASELINE.RL.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = self._config_baseline.BASELINE.RL.SLACK_REWARD

        current_target_distance = self._distance_target()
        reward += self._previous_target_distance - current_target_distance
        self._previous_target_distance = current_target_distance

        if self._episode_success():
            reward += self._config_baseline.BASELINE.RL.SUCCESS_REWARD

        return reward

    def _distance_target(self):
        current_position = self._env.sim.get_agent_state().position.tolist()
        target_position = self._env.current_episode.goals[0].position
        distance = self._env.sim.geodesic_distance(
            current_position, target_position
        )
        return distance

    def _episode_success(self):
        # challenge API
        # if (
        #     self._previous_action
        #     == SIM_NAME_TO_ACTION[SimulatorActions.STOP.value]
        #     and self._distance_target() < self._config_env.SUCCESS_DISTANCE
        # ):
        #     return True
        if (
            self._previous_action
            == name2action[SimulatorActions.STOP.name]
            and self._distance_target() < self._config_env.SUCCESS_DISTANCE
        ):
            return True
        return False

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        info = {}

        if self.get_done(observations):
            info["spl"] = self.habitat_env.get_metrics()["spl"]

        return info


def make_env_fn(config_env, config_baseline, rank):
    dataset = PointNavDatasetV1(config_env.DATASET)
    config_env.defrost()
    config_env.SIMULATOR.SCENE = dataset.episodes[0].scene_id  # data/scene_datasets/gibson/Cantwell.glb
    config_env.freeze()
    env = NavRLEnv(
        config_env=config_env, config_baseline=config_baseline, dataset=dataset
    )
    env.seed(rank)
    return env


def construct_envs(args):
    config_env = cfg_env(args.task_config)
    config_env.defrost()

    agent_sensors = config_env.SIMULATOR.AGENT_0.SENSORS
    for sensor in agent_sensors:
        assert sensor in ["RGB_SENSOR", "DEPTH_SENSOR"]
    config_env.SIMULATOR.AGENT_0.SENSORS = agent_sensors
    config_env.freeze()
    # env_configs.append(config_env)

    config_baseline = cfg_baseline()

    logger.info("config_env: {}".format(config_env))

    envs = make_env_fn(config_env=config_env, config_baseline=config_baseline, rank=0)

    goal_radius = envs.episodes[0].goals[0].radius
    if goal_radius is None:
        goal_radius = config_env.SIMULATOR.FORWARD_STEP_SIZE
    follower = ShortestPathFollower(envs.habitat_env.sim, goal_radius, False)
    follower.mode = "geodesic_path"  # "greedy"

    return envs, follower


def main():
    parser = ppo_args()
    args = parser.parse_args()

    random.seed(args.seed)

    device = torch.device("cuda:{}".format(args.pth_gpu_id))

    logger.add_filehandler(args.log_file)
    writer = SummaryWriter(args.tb_log_dir)

    if not os.path.isdir(args.checkpoint_folder):
        os.makedirs(args.checkpoint_folder)

    for p in sorted(list(vars(args))):
        logger.info("{}: {}".format(p, getattr(args, p)))

    envs, follower = construct_envs(args)

    actor_critic = Policy(
        observation_space=envs.observation_space,
        action_space=envs.action_space,
        hidden_size=args.hidden_size,
    )
    actor_critic.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    logger.info(
        "agent number of parameters: {}".format(
            sum(param.numel() for param in actor_critic.parameters())
        )
    )


    t_start = time()
    env_time = 0
    pth_time = 0
    count_steps = 0
    count_checkpoints = 0

    optimizer = optim.Adam(actor_critic.parameters(), lr=args.lr, eps=args.eps)
    done = True
    num_updates = 0
    total_rewards = 0
    episode_counts = 0
    episode_spls = 0
    total_action_loss = 0
    for update in range(args.num_updates):
        episode_rewards = 0

        if args.use_linear_lr_decay:
            update_linear_schedule(
                optimizer, update, args.num_updates, args.lr
            )

        # agent.clip_param = args.clip_param * (1 - update / args.num_updates)

        observations = envs.reset()
        observations = [observations]
        batch = batch_obs(observations, device)
        if done:
            recurrent_hidden_states = torch.zeros(
                1, 1, args.hidden_size
            ).to(device)
        masks = torch.ones(1, 1, 1).to(device)
        recurrent_hidden_states = torch.tensor(recurrent_hidden_states.detach().cpu().numpy()).to(device)
        # print('Env reset!')

        probs = []
        st_actions = []
        done = False
        while not done:
            t_sample_action = time()
            # sample actions

            (
                distribution,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = actor_critic.act(
                batch,
                recurrent_hidden_states[0],
                masks[0],
                deterministic=True
            )
            recurrent_hidden_states = recurrent_hidden_states.unsqueeze(0)
            pth_time += time() - t_sample_action

            t_step_env = time()

            shortest_path_action = follower.get_next_action(
                envs.habitat_env.current_episode.goals[0].position
            )
            shortest_path_action = name2action[shortest_path_action.name]
            outputs = envs.step(actions.item())
            # print(actions.item(), shortest_path_action)
            observations, rewards, done, infos = outputs
            print(actions.item(), done)
            env_time += time() - t_step_env

            t_update_stats = time()
            observations = [observations]
            batch = batch_obs(observations, device)
            rewards = rewards

            episode_rewards += rewards

            count_steps += 1

            probs.append(distribution.probs.squeeze(0))
            st_actions.append(torch.tensor([shortest_path_action]).squeeze(0))
            pth_time += time() - t_update_stats

            if done:
                episode_spls += infos["spl"]
                episode_counts += 1
                break

        t_update_model = time()
        total_rewards += episode_rewards
        avg_rewards = total_rewards / episode_counts
        avg_episode_spls = episode_spls / episode_counts

        prob_tensor = torch.stack(probs).to(device)
        st_actions_tensor = torch.stack(st_actions).to(device)
        action_loss = criterion(prob_tensor, st_actions_tensor) * 1e4
        # print(prob_tensor.shape)
        optimizer.zero_grad()
        action_loss.backward()
        nn.utils.clip_grad_norm_(
            actor_critic.parameters(), args.max_grad_norm
        )

        optimizer.step()

        total_action_loss += action_loss.item()

        num_updates += len(st_actions_tensor)

        action_loss_avg = total_action_loss / num_updates

        pth_time += time() - t_update_model

        # log stats
        if update > 0 and update % args.log_interval == 0:
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    update, count_steps / (time() - t_start)
                )
            )

            logger.info(
                "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                "frames: {} action_loss_avg: {:3f} avg_rewards: {:3f} avg_episode_spls: {:3f}".format(update, env_time, pth_time, count_steps, action_loss_avg, avg_rewards, avg_episode_spls)
            )

        # checkpoint model
        if update % args.checkpoint_interval == 0:
            checkpoint = {"state_dict": actor_critic.state_dict()}
            torch.save(
                checkpoint,
                os.path.join(
                    args.checkpoint_folder,
                    "ckpt.{}.pth".format(count_checkpoints),
                ),
            )
            count_checkpoints += 1

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()

if __name__ == "__main__":
    main()
