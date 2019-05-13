#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

'''
python collect_data.py --sim-gpu-id 0 --num-processes 1
'''
import sys
sys.path.insert(0, './map_and_plan_agent/')
import argparse
import os

import habitat
from habitat.config.default import get_config
from config.default import cfg as cfg_baseline
from habitat.sims.habitat_simulator import SimulatorActions, SIM_NAME_TO_ACTION
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
import numpy as np
import cv2

# from map_and_plan_agent.slam import DepthMapperAndPlanner
from map_and_plan_agent.slam_custom import DepthMapperAndPlanner

class NavRLEnv(habitat.RLEnv):
    def __init__(self, config_env, config_baseline, dataset):
        self._config_env = config_env.TASK
        self._config_baseline = config_baseline
        self._previous_target_distance = None
        self._previous_action = None
        self._episode_distance_covered = None
        super().__init__(config_env, dataset)
        self.scene_id = None

    def reset(self):
        self._previous_action = None

        observations = super().reset()

        self._previous_target_distance = self.habitat_env.current_episode.info[
            "geodesic_distance"
        ]

        return observations, self.habitat_env.current_episode.scene_id

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
        if (
                self._previous_action
                == SIM_NAME_TO_ACTION[SimulatorActions.STOP.value]
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
    config_env.ENVIRONMENT.MAX_EPISODE_STEPS = 2000
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
        default=[0],
        help="gpu id on which scenes are loaded",
    )
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--count-test-episodes", type=int, default=88)
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
        default="tasks/pointnav_gibson_rgbd.yaml",
        help="path to config yaml containing information about task",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="data/habitat_training_data_tmp",
        help="directory to save result",
    )
    parser.add_argument(
        "--random-agent",
        action="store_true",
        default=False,
        help="use random agent",
    )
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    config_env = get_config(config_file=args.task_config)
    config_env.defrost()

    agent_sensors = config_env.SIMULATOR.AGENT_0.SENSORS

    for sensor in agent_sensors:
        assert sensor in ["RGB_SENSOR", "DEPTH_SENSOR"]
    config_env.freeze()

    config_baseline = cfg_baseline()

    envs = make_env_fn(config_env=config_env, config_baseline=config_baseline, rank=0, episodes_index=0)
    if not args.random_agent:
        agent = DepthMapperAndPlanner(map_size_cm=1200, out_dir=args.outdir, mark_locs=True,
                                      reset_if_drift=True, count=-1, close_small_openings=True,
                                      recover_on_collision=True, fix_thrashing=True, goal_f=1.1, point_cnt=2)

    test_episodes = 0
    prev_scene_id = None
    total_counter = -1
    if args.random_agent:
        while test_episodes < args.count_test_episodes:
            observations, scene_id = envs.reset()
            total_counter += 1
            if total_counter != 577:
                print(total_counter)
                continue
            # if scene_id == prev_scene_id:
            #     continue
            prev_scene_id = scene_id
            episode_counter = 0
            cv2.imwrite(os.path.join(args.outdir, "rgb_{0:02d}_{1:07d}.png".format(test_episodes, episode_counter)), observations['rgb'])
            np.save(os.path.join(args.outdir, "depth_{0:02d}_{1:07d}".format(test_episodes, episode_counter)), observations['depth'])
            if not args.random_agent:
                agent.reset()
            while episode_counter < 1000:
                episode_counter = episode_counter + 1
                if args.random_agent:
                    actions = np.random.randint(3)
                else:
                    actions = agent.act(observations=observations)
                    actions = np.random.randint(3)

                observations, rewards, dones, infos = envs.step(actions)
                # observations: [{'rgb': array([...], dtype=uint8)}, {'depth': array([...], dtype=float32)}, 'pointgoal': array([5.6433434, 2.70739  ], dtype=float32)}]

                cv2.imwrite(os.path.join(args.outdir, "rgb_{0:02d}_{1:07d}.png".format(test_episodes, episode_counter)), observations['rgb'])
                np.save(os.path.join(args.outdir, "depth_{0:02d}_{1:07d}".format(test_episodes, episode_counter)), observations['depth'])

            print('Episode {0}:'.format(test_episodes))
            test_episodes += 1
    else:
        while test_episodes < 1000:
            observations, scene_id = envs.reset()
            total_counter += 1
            if total_counter != 577 and total_counter != 578:
                print(total_counter)
                continue
            # if scene_id == prev_scene_id:
            #     continue
            prev_scene_id = scene_id
            episode_counter = 0
            # cv2.imwrite(os.path.join(args.outdir, "rgb_{0:02d}_{1:07d}.png".format(test_episodes, episode_counter)),
            #             observations['rgb'])
            # np.save(os.path.join(args.outdir, "depth_{0:02d}_{1:07d}".format(test_episodes, episode_counter)),
            #         observations['depth'])
            if not args.random_agent:
                agent.reset()
            while episode_counter < 500:
                episode_counter = episode_counter + 1
                if args.random_agent:
                    actions = np.random.randint(3)
                else:
                    actions = agent.act(observations=observations)
                    actions = np.random.randint(3)

                observations, rewards, dones, infos = envs.step(actions)
                # observations: [{'rgb': array([...], dtype=uint8)}, {'depth': array([...], dtype=float32)}, 'pointgoal': array([5.6433434, 2.70739  ], dtype=float32)}]

                # cv2.imwrite(os.path.join(args.outdir, "rgb_{0:02d}_{1:07d}.png".format(test_episodes, episode_counter)),
                #             observations['rgb'])
                # np.save(os.path.join(args.outdir, "depth_{0:02d}_{1:07d}".format(test_episodes, episode_counter)),
                #         observations['depth'])

            print('Episode {0}:'.format(test_episodes))
            test_episodes += 1

    print('Eval Finished!')


if __name__ == "__main__":
    main()
