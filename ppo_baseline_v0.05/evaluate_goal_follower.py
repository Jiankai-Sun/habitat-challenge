#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
sys.path.insert(0, './map_and_plan_agent/')
import argparse
import os
import torch

import habitat
from habitat.config.default import get_config
from config.default import cfg as cfg_baseline
# challenge API
# from habitat.sims.habitat_simulator import SimulatorActions, SIM_NAME_TO_ACTION
from habitat.sims.habitat_simulator import SimulatorActions
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
import numpy as np

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
        #         self._previous_action
        #         == SIM_NAME_TO_ACTION[SimulatorActions.STOP.name]
        #         and self._distance_target() < self._config_env.SUCCESS_DISTANCE
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


def make_env_fn(config_env, config_baseline, rank, episodes_index=0):
    dataset = PointNavDatasetV1(config_env.DATASET)
    config_env.defrost()
    config_env.SIMULATOR.SCENE = dataset.episodes[episodes_index].scene_id  # data/scene_datasets/gibson/Cantwell.glb
    config_env.freeze()
    env = NavRLEnv(
        config_env=config_env, config_baseline=config_baseline, dataset=dataset
    )
    env.seed(rank)
    return env

PAUSE_TIME = 100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None, type=str)
    parser.add_argument(
        "--sim-gpu-id",
        nargs='+',
        type=int,
        default=0,
        help="gpu id on which scenes are loaded",
    )
    parser.add_argument("--pth-gpu-id", type=int, default=0)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--count-test-episodes", type=int, default=1000)
    parser.add_argument(
        "--sensors",
        type=str,
        default="RGB_SENSOR,DEPTH_SENSOR",
        help="comma separated string containing different"
        "sensors to use, currently 'RGB_SENSOR' and"
        "'DEPTH_SENSOR' are supported",
    )
    parser.add_argument(
        "--task-config",
        type=str,
        default="tasks/pointnav_gibson_rgb.yaml",
        help="path to config yaml containing information about task",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="data/slam_result/goal_follower",
        help="directory to save result",
    )

    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    device = torch.device("cuda:{}".format(args.pth_gpu_id))
    # challenge API
    # config_env = get_config(config_file=args.task_config)
    config_env = get_config('configs/tasks/pointnav_gibson_rgb.yaml')

    config_env.defrost()
    config_env.DATASET.SPLIT = "val"

    agent_sensors = config_env.SIMULATOR.AGENT_0.SENSORS

    for sensor in agent_sensors:
        assert sensor in ["RGB_SENSOR", "DEPTH_SENSOR"]
    config_env.freeze()

    config_baseline = cfg_baseline()

    envs = make_env_fn(config_env=config_env, config_baseline=config_baseline, rank=0, episodes_index=0)

    # print(dir(envs.habitat_env.sim))

    goal_radius = envs.episodes[0].goals[0].radius
    if goal_radius is None:
        goal_radius = config_env.SIMULATOR.FORWARD_STEP_SIZE
    agent = ShortestPathFollower(envs.habitat_env.sim, goal_radius, False)
    agent.mode = "geodesic_path"  # "greedy"

    episode_rewards = torch.zeros(1, 1, device=device)
    episode_spls = torch.zeros(1, 1, device=device)
    episode_success = torch.zeros(1, 1, device=device)
    episode_counts = torch.zeros(1, 1, device=device)
    current_episode_reward = torch.zeros(1, 1, device=device)

    test_episodes = 0
    spl_np = np.zeros((1000, 3))
    while test_episodes < args.count_test_episodes:
        observations = envs.reset()

        dones = False

        while not dones:
            actions = agent.get_next_action(
                envs.habitat_env.current_episode.goals[0].position
            )

            outputs = envs.step(name2action[actions.name])

            # observations: [{'rgb': array([...], dtype=uint8)}, {'depth': array([...], dtype=float32)}, 'pointgoal': array([5.6433434, 2.70739  ], dtype=float32)}]
            observations, rewards, dones, infos = outputs

            not_done_masks = torch.tensor(
                [0.0 if dones else 1.0],
                dtype=torch.float,
                device=device,
            )

            for i in range(not_done_masks.shape[0]):
                if not_done_masks[i].item() == 0:
                    episode_spls[i] += infos["spl"]
                    spl_record = infos["spl"]
                    if infos["spl"] > 0:
                        episode_success[i] += 1

            rewards = torch.tensor(
                [rewards], dtype=torch.float, device=device
            ).unsqueeze(1)
            current_episode_reward += rewards
            episode_rewards += (1 - not_done_masks) * current_episode_reward
            episode_counts += 1 - not_done_masks
            current_episode_reward_data = current_episode_reward.item()
            current_episode_reward *= not_done_masks

        episode_reward_mean = (episode_rewards / episode_counts).mean().item()
        episode_spl_mean = (episode_spls / episode_counts).mean().item()
        episode_success_mean = (episode_success / episode_counts).mean().item()

        print('Episode {0}:'.format(test_episodes))
        print("Average episode reward: {:.6f}".format(episode_reward_mean))
        print("Average episode success: {:.6f}".format(episode_success_mean))
        print("Average episode spl: {:.6f}".format(episode_spl_mean))
        print("Episode reward: {:.6f}".format(current_episode_reward_data))
        print("Episode success: {0}".format(infos["spl"] > 0))
        print("Episode spl: {:.6f}".format(infos["spl"]))
        spl_np[test_episodes, 0] = test_episodes
        spl_np[test_episodes, 1] = infos["spl"]
        spl_np[test_episodes, 2] = episode_spl_mean

        test_episodes += 1
        np.savetxt(os.path.join(args.outdir, 'spls.txt'), spl_np)
    print('Eval Finished!')


if __name__ == "__main__":
    main()
