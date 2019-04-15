#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import torch

import habitat
from habitat.config.default import get_config
from config.default import cfg as cfg_baseline

from train_ppo import make_env_fn
from rl.ppo.ppo_alg import PPO
from rl.ppo.policy import Policy
from rl.ppo.ppo_utils import batch_obs
import sys

import numpy as np
import cv2
import os
import subprocess
import shutil
# import matplotlib.pyplot as plt
from map_and_plan_agent.slam import DepthMapperAndPlanner as agent

def visualize_traj(env, traj_coors, pointgoal, output_file):
    """
    env: interative environment;
    traj_coors: list of each points' coordinate on the trajectory (coordinates are in the format of (x, _, y));
    output_dir: output directory of the topdown freespace map
    """
    output_dir = os.path.join(*(output_file.split('/')[:-1]))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    topdown_freespace_map = np.uint8(np.zeros([512, 512, 3]))
    pix_f = [255, 255, 255]

    minx = 100
    miny = 100
    maxx = -100
    maxy = -100

    for _ in range(100000):
        x, _, y = env.habitat_env.sim.sample_navigable_point()
        if x < minx:
            minx = x
        if y < miny:
            miny = y
        if x > maxx:
            maxx = x
        if y > maxy:
            maxy = y

    for _ in range(100000):
        x, _, y = env.habitat_env.sim.sample_navigable_point()
        gx = int((x - minx) / (maxx - minx) * 511)
        gy = int((y - miny) / (maxy - miny) * 511)
        topdown_freespace_map[gx, gy] = pix_f

    for i, coor in enumerate(traj_coors):
        x, _, y = coor
        gx = int((x - minx) / (maxx - minx) * 511)
        gy = int((y - miny) / (maxy - miny) * 511)
        if i == 0:
            # print(gy, gx)
            cv2.circle(topdown_freespace_map, (gy, gx), 8, (0, 126, 255), -1)
        elif i == len(traj_coors) - 1:
            cv2.circle(topdown_freespace_map, (gy, gx), 8, (255, 126, 0), -1)
        else:
            cv2.circle(topdown_freespace_map, (gy, gx), 6, (0, 255, 0), -1)

    gx = int((pointgoal[0] - minx) / (maxx - minx) * 511)
    gy = int((pointgoal[2] - miny) / (maxy - miny) * 511)

    cv2.circle(topdown_freespace_map, (gy, gx), 8, (0, 0, 255), -1)

    cv2.imwrite(os.path.join(output_file), topdown_freespace_map)

def frame_to_video(fileloc, t_w, t_h, destination):
    command = ['ffmpeg',
               '-y',
               '-loglevel', 'fatal',
               '-framerate', '4',
               '-f', 'image2',  # 'image2pipe',
               '-i', fileloc,
               '-vcodec', 'libx264',
               '-pix_fmt', 'yuv420p',
               destination]
    # print(command)
    ffmpeg = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = ffmpeg.communicate()
    if err:
        print('error', err)
        return None

    video = np.fromstring(out, dtype='uint8').reshape((-1, t_h, t_w, 3))  #NHWC
    return video

PAUSE_TIME = 100

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    # parser.add_argument("--sim-gpu-id", type=int, required=True)
    parser.add_argument(
        "--sim-gpu-id",
        nargs='+',
        type=int,
        required=True,
        default=[0],
        help="gpu id on which scenes are loaded",
    )
    parser.add_argument("--pth-gpu-id", type=int, required=True)
    parser.add_argument("--num-processes", type=int, required=True)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--count-test-episodes", type=int, default=100)
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
        default="tasks/pointnav.yaml",
        help="path to config yaml containing information about task",
    )
    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.pth_gpu_id))

    # env_configs = []
    # baseline_configs = []

    config_env = get_config(config_file=args.task_config)
    config_env.defrost()
    config_env.DATASET.SPLIT = "val"
    # TODO
    # config_env.SIMULATOR.RGB_SENSOR.HFOV = 180
    # config_env.SIMULATOR.DEPTH_SENSOR.HFOV = 180

    agent_sensors = config_env.SIMULATOR.AGENT_0.SENSORS
    if 'RGB_SENSOR' in agent_sensors and not 'DEPTH_SENSOR' in agent_sensors:
        img_width = 90 * 2
        img_height = 90
    elif 'RGB_SENSOR' in agent_sensors and 'DEPTH_SENSOR' in agent_sensors:
        img_width = 90 * 3
        img_height = 90
    else:
        raise NotImplementedError('Unsupported mode in observations')

    # print('agent_sensors: ', agent_sensors)
    for sensor in agent_sensors:
        assert sensor in ["RGB_SENSOR", "DEPTH_SENSOR"]
    config_env.freeze()
    # env_configs.append(config_env)

    config_baseline = cfg_baseline()
    # baseline_configs.append(config_baseline)
    #
    # assert len(baseline_configs) > 0, "empty list of datasets"

    envs = make_env_fn(config_env, config_baseline, 0)  # habitat.Env(env_configs)

    ckpt = torch.load(args.model_path, map_location=device)

    episode_rewards = torch.zeros(1, 1, device=device)
    episode_spls = torch.zeros(1, 1, device=device)
    episode_success = torch.zeros(1, 1, device=device)
    episode_counts = torch.zeros(1, 1, device=device)
    current_episode_reward = torch.zeros(1, 1, device=device)

    not_done_masks = torch.zeros(args.num_processes, 1, device=device)

    action_times = 0
    video_folder_index = 0

    if not os.path.exists(os.path.join("data", "video")):
        os.makedirs(os.path.join("data", "video"))
    success_log = open(os.path.join("data", "video", "success_log.txt"), "w")

    while video_folder_index < args.count_test_episodes:
        observations = envs.reset()
        observations = [observations]
        batch = batch_obs(observations)
        for sensor in batch:
            batch[sensor] = batch[sensor].to(device)

        dones = False
        target_position = envs._env.current_episode.goals[0].position
        # print('target_position: ', target_position)
        traj_coors = []

        while not dones:
            current_position = envs._env.sim.get_agent_state().position.tolist()
            # print('current_position: ', current_position)
            traj_coors.append(current_position)

            if not os.path.exists(os.path.join("data", "video", str(video_folder_index))):
                os.makedirs(os.path.join("data", "video", str(video_folder_index)))
            action_times += 1
            actions = agent.act(batch)

            outputs = envs.step(actions.item())

            # observations: [{'rgb': array([...], dtype=uint8)}, {'depth': array([...], dtype=float32)}, 'pointgoal': array([5.6433434, 2.70739  ], dtype=float32)}]
            observations, rewards, dones, infos = outputs

            observations = [observations]

            batch = batch_obs(observations)
            for sensor in batch:
                batch[sensor] = batch[sensor].to(device)

            not_done_masks = torch.tensor(
                [0.0 if dones else 1.0],
                dtype=torch.float,
                device=device,
            )
            for i in range(not_done_masks.shape[0]):
                if not_done_masks[i].item() == 0:
                    episode_spls[i] += infos["spl"]

                    action_times = 0

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


if __name__ == "__main__":
    main()
