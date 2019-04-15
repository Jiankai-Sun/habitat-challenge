#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import argparse
from math import pi

import numpy as np
import habitat
from habitat.config.default import get_config
from config.default import cfg as cfg_baseline

from train_ppo import make_env_fn
from rl.ppo.ppo_utils import batch_obs
import torch

from habitat.sims.habitat_simulator import (
    SimulatorActions,
    SIM_ACTION_TO_NAME,
    SIM_NAME_TO_ACTION,
)

NON_STOP_ACTIONS = [
    k
    for k, v in SIM_ACTION_TO_NAME.items()
    if v != SimulatorActions.STOP.value
]


class RandomAgent(habitat.Agent):
    def __init__(self, success_distance):
        self.dist_threshold_to_stop = success_distance

    def reset(self):
        pass

    def is_goal_reached(self, observations):
        dist = observations["pointgoal"][0]
        return dist <= self.dist_threshold_to_stop

    def act(self, observations):
        if self.is_goal_reached(observations):
            action = SIM_NAME_TO_ACTION[SimulatorActions.STOP.value]
        else:
            action = np.random.choice(NON_STOP_ACTIONS)
        return action


class ForwardOnlyAgent(RandomAgent):
    def act(self, observations):
        if self.is_goal_reached(observations):
            action = SIM_NAME_TO_ACTION[SimulatorActions.STOP.value]
        else:
            action = SIM_NAME_TO_ACTION[SimulatorActions.FORWARD.value]
        return action


class RandomForwardAgent(RandomAgent):
    def __init__(self, success_distance):
        super().__init__(success_distance)
        self.FORWARD_PROBABILITY = 0.8

    def act(self, observations):
        if self.is_goal_reached(observations):
            action = SIM_NAME_TO_ACTION[SimulatorActions.STOP.value]
        else:
            if np.random.uniform(0, 1, 1) < self.FORWARD_PROBABILITY:
                action = SIM_NAME_TO_ACTION[SimulatorActions.FORWARD.value]
            else:
                action = np.random.choice(
                    [
                        SIM_NAME_TO_ACTION[SimulatorActions.LEFT.value],
                        SIM_NAME_TO_ACTION[SimulatorActions.RIGHT.value],
                    ]
                )

        return action


class GoalFollower(RandomAgent):
    def __init__(self, success_distance):
        super().__init__(success_distance)
        self.pos_th = self.dist_threshold_to_stop
        self.angle_th = float(np.deg2rad(15))
        self.random_prob = 0

    def normalize_angle(self, angle):
        if angle < -pi:
            angle = 2.0 * pi + angle
        if angle > pi:
            angle = -2.0 * pi + angle
        return angle

    def turn_towards_goal(self, angle_to_goal):
        if angle_to_goal > pi or (
            (angle_to_goal < 0) and (angle_to_goal > -pi)
        ):
            action = SIM_NAME_TO_ACTION[SimulatorActions.RIGHT.value]
        else:
            action = SIM_NAME_TO_ACTION[SimulatorActions.LEFT.value]
        return action

    def act(self, observations):
        if self.is_goal_reached(observations):
            action = SIM_NAME_TO_ACTION[SimulatorActions.STOP.value]
        else:
            angle_to_goal = self.normalize_angle(
                np.array(observations["pointgoal"][1])
            )
            if abs(angle_to_goal) < self.angle_th:
                action = SIM_NAME_TO_ACTION[SimulatorActions.FORWARD.value]
            else:
                action = self.turn_towards_goal(angle_to_goal)

        return action


def get_all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in get_all_subclasses(c)]
    )


def get_agent_cls(agent_class_name):
    sub_classes = [
        sub_class
        for sub_class in get_all_subclasses(habitat.Agent)
        if sub_class.__name__ == agent_class_name
    ]
    return sub_classes[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--success-distance", type=float, default=0.2)
    parser.add_argument(
        "--task-config", type=str, default="tasks/pointnav_gibson_rgbd.yaml"
    )
    parser.add_argument("--agent-class", type=str, default="GoalFollower")
    args = parser.parse_args()

    device = torch.device("cuda:{}".format(7))
    agent = get_agent_cls(args.agent_class)(
        success_distance=args.success_distance
    )

    config_env = get_config(config_file=args.task_config)
    config_env.defrost()
    print('config_env.DATASET.SPLIT: ', config_env.DATASET.SPLIT)
    config_baseline = cfg_baseline()
    envs = make_env_fn(config_env, config_baseline, 0)  # habitat.Env(env_configs)

    video_folder_index = 0
    episode_rewards = torch.zeros(1, 1, device=device)
    episode_spls = torch.zeros(1, 1, device=device)
    episode_success = torch.zeros(1, 1, device=device)
    episode_counts = torch.zeros(1, 1, device=device)
    current_episode_reward = torch.zeros(1, 1, device=device)

    observations_dict = {'rgb': [], 'depth': [], 'pointgoal': []}
    while video_folder_index < 70 * 7e4:
        video_folder_index += 1
        observations = envs.reset()
        for sensor in observations:
            observations_dict[sensor].append(observations[sensor])
        # print('observations.keys(): ', observations.keys())  # dict_keys(['rgb', 'depth', 'pointgoal'])

        dones = False

        while not dones:
            action = agent.act(observations)

            outputs = envs.step(action)

            observations, rewards, dones, infos = outputs

            for sensor in observations:
                observations_dict[sensor].append(observations[sensor])

            not_done_masks = torch.tensor(
                [0.0 if dones else 1.0],
                dtype=torch.float,
                device=device,
            )

            rewards = torch.tensor(
                [rewards], dtype=torch.float, device=device
            ).unsqueeze(1)
            current_episode_reward += rewards
            episode_rewards += (1 - not_done_masks) * current_episode_reward
            episode_counts += 1 - not_done_masks
            current_episode_reward *= not_done_masks
        episode_reward_mean = (episode_rewards / episode_counts).mean().item()
        episode_spl_mean = (episode_spls / episode_counts).mean().item()
        episode_success_mean = (episode_success / episode_counts).mean().item()

        print('Episode {0}:'.format(video_folder_index))
        print("Average episode reward: {:.6f}".format(episode_reward_mean))
        print("Average episode success: {:.6f}".format(episode_success_mean))
        print("Average episode spl: {:.6f}".format(episode_spl_mean))

        rgb_mean = np.array(observations_dict['rgb']).mean(axis=(0, 1, 2)) / 255
        rgb_std = np.array(observations_dict['rgb']).std(axis=(0, 1, 2)) / 255
        depth_mean = np.array(observations_dict['depth']).mean(axis=(0, 1, 2)) / 255
        depth_std = np.array(observations_dict['depth']).std(axis=(0, 1, 2)) / 255
        print('rgb_mean: {0}, rgb_std: {1}, depth_mean: {2}, depth_std: {3}'.format(rgb_mean, rgb_std, depth_mean, depth_std))
        np.save('mean_std.npy', [rgb_mean, rgb_std, depth_mean, depth_std])

        '''
        # Convert back
        convert_image1 = image.numpy()
        convert_image1 = np.squeeze(convert_image1)  # 3* 224 *224, C * H * W
        convert_image1 = convert_image1 * np.reshape(std_file, (3, 1, 1)) + np.reshape(mean_file, (3, 1, 1))
        convert_image1 = np.transpose(convert_image1, (1, 2, 0))  # H * W * C
        print(convert_image1.shape)

        convert_image1 = convert_image1 * 255
        '''

if __name__ == "__main__":
    main()