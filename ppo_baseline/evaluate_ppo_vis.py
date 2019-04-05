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
from rl.ppo import PPO, Policy
from rl.ppo.utils import batch_obs

import numpy as np
import cv2
import os
import subprocess
import shutil

def visualize_traj(env, traj_coors, output_dir):
    """
    env: interative environment;
    traj_coors: list of each points' coordinate on the trajectory (coordinates are in the format of (x, _, y));
    output_dir: output directory of the topdown freespace map
    """
    topdown_freespace_map = np.uint8(np.zeros([512, 512, 3]))
    pix_f = [255, 255, 255]

    minx = 100
    miny = 100
    maxx = -100
    maxy = -100

    for _ in range(10000):
        x, _, y = env.sim.sample_navigable_point()
        if x < minx:
            minx = x
        if y < miny:
            miny = y
        if x > maxx:
            maxx = x
        if y > maxy:
            maxy = y

    for _ in range(100000):
        x, _, y = env.sim.sample_navigable_point()
        gx = int((x - minx) / (maxx - minx) * 511)
        gy = int((y - miny) / (maxy - miny) * 511)
        topdown_freespace_map[gx, gy] = pix_f

    for i, coor in enumerate(traj_coors):
        x, _, y = coor
        gx = int((x - minx) / (maxx - minx) * 511)
        gy = int((y - miny) / (maxy - miny) * 511)
        if i == 0:
            print(gx, gy)
            cv2.circle(topdown_freespace_map, (gx, gy), 8, (0, 126, 255), -1)
        elif i == len(traj_coors) - 1:
            cv2.circle(topdown_freespace_map, (gx, gy), 8, (255, 126, 0), -1)
        else:
            cv2.circle(topdown_freespace_map, (gx, gy), 6, (0, 255, 0), -1)
    cv2.imwrite(os.path.join(output_dir, 'example_topdown_freespace.png'), topdown_freespace_map)

def frame_to_video(fileloc, t_w, t_h, destination):
    command = ['ffmpeg',
               '-y',
               '-loglevel', 'fatal',
               '-framerate', '1',
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--sim-gpu-id", type=int, required=True)
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

    env_configs = []
    baseline_configs = []

    for _ in range(args.num_processes):
        config_env = get_config(config_file=args.task_config)
        config_env.defrost()
        config_env.DATASET.SPLIT = "val"

        agent_sensors = args.sensors.strip().split(",")
        for sensor in agent_sensors:
            assert sensor in ["RGB_SENSOR", "DEPTH_SENSOR"]
        config_env.SIMULATOR.AGENT_0.SENSORS = agent_sensors
        config_env.freeze()
        env_configs.append(config_env)

        config_baseline = cfg_baseline()
        baseline_configs.append(config_baseline)

    assert len(baseline_configs) > 0, "empty list of datasets"

    envs = habitat.VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(
                zip(env_configs, baseline_configs, range(args.num_processes))
            )
        ),
    )

    ckpt = torch.load(args.model_path, map_location=device)

    actor_critic = Policy(
        observation_space=envs.observation_spaces[0],  # depth:Box(256, 256, 1), rgb:Box(256, 256, 3), pointgoal:Box(2,)
        action_space=envs.action_spaces[0],  # Discrete(4)
        hidden_size=512,
    )

    print('envs.observation_spaces[0]: ', envs.observation_spaces, 'envs.action_spaces[0]: ', envs.action_spaces)
    actor_critic.to(device)

    ppo = PPO(
        actor_critic=actor_critic,
        clip_param=0.1,
        ppo_epoch=4,
        num_mini_batch=32,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        lr=2.5e-4,
        eps=1e-5,
        max_grad_norm=0.5,
    )

    ppo.load_state_dict(ckpt["state_dict"])

    actor_critic = ppo.actor_critic

    observations = envs.reset()
    batch = batch_obs(observations)
    for sensor in batch:
        batch[sensor] = batch[sensor].to(device)

    episode_rewards = torch.zeros(envs.num_envs, 1, device=device)
    episode_spls = torch.zeros(envs.num_envs, 1, device=device)
    episode_success = torch.zeros(envs.num_envs, 1, device=device)
    episode_counts = torch.zeros(envs.num_envs, 1, device=device)
    current_episode_reward = torch.zeros(envs.num_envs, 1, device=device)

    test_recurrent_hidden_states = torch.zeros(
        args.num_processes, args.hidden_size, device=device
    )
    not_done_masks = torch.zeros(args.num_processes, 1, device=device)

    action_times = 0
    video_folder_index = 0
    traj_coors = []
    while episode_counts.sum() < args.count_test_episodes:
        if not os.path.exists(os.path.join("data", "video", str(video_folder_index))):
            os.makedirs(os.path.join("data", "video", str(video_folder_index)))
        action_times += 1

        with torch.no_grad():
            _, actions, _, test_recurrent_hidden_states = actor_critic.act(
                batch,
                test_recurrent_hidden_states,
                not_done_masks,
                deterministic=False,
            )

        outputs = envs.step([a[0].item() for a in actions])

        # observations: [{'rgb': array([...], dtype=uint8)}, {'depth': array([...], dtype=float32)}, 'pointgoal': array([5.6433434, 2.70739  ], dtype=float32)}]
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        mat = np.array(cv2.resize(observations[0]['rgb'], (480, 360)))
        cv2.putText(mat, "action:" + str(actions[0].item()) + " reward:" + str(rewards[0]),
                    # + " target:" + str(observations[0]['pointgoal'].tolist()),
                    (15, 15),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("Habitat-API Evaluation", mat)
        cv2.moveWindow("Habitat-API Evaluation", 0, 0)
        cv2.waitKey(200)
        cv2.imwrite(os.path.join("data", "video", str(video_folder_index), str(action_times) + ".png"),
                    observations[0]['rgb'])  # (90, 120, 3)

        batch = batch_obs(observations)
        for sensor in batch:
            batch[sensor] = batch[sensor].to(device)

        not_done_masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=device,
        )

        if not_done_masks[0] == 0.0:
            cv2.putText(mat, 'Done!', (160, 200), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Habitat-API Evaluation", mat)
            cv2.moveWindow("Habitat-API Evaluation", 0, 0)
            cv2.waitKey(400)
            cv2.destroyAllWindows()
            frame_to_video(fileloc=os.path.join("data", "video", str(video_folder_index), "%d.png"), t_w=120,
                           t_h=90,
                           destination=os.path.join("data", "video", str(video_folder_index)+".mp4"))
            shutil.rmtree(os.path.join("data", "video", str(video_folder_index)))
            action_times = 0
            video_folder_index += 1
            visualize_traj(envs, traj_coors, './visualize')
            traj_coors = []


        for i in range(not_done_masks.shape[0]):
            if not_done_masks[i].item() == 0:
                episode_spls[i] += infos[i]["spl"]
                if infos[i]["spl"] > 0:
                    episode_success[i] += 1

        rewards = torch.tensor(
            rewards, dtype=torch.float, device=device
        ).unsqueeze(1)
        current_episode_reward += rewards
        episode_rewards += (1 - not_done_masks) * current_episode_reward
        episode_counts += 1 - not_done_masks
        current_episode_reward *= not_done_masks

    episode_reward_mean = (episode_rewards / episode_counts).mean().item()
    episode_spl_mean = (episode_spls / episode_counts).mean().item()
    episode_success_mean = (episode_success / episode_counts).mean().item()

    print("Average episode reward: {:.6f}".format(episode_reward_mean))
    print("Average episode success: {:.6f}".format(episode_success_mean))
    print("Average episode spl: {:.6f}".format(episode_spl_mean))


if __name__ == "__main__":
    main()
