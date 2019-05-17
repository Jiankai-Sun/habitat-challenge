#!/usr/bin/env bash

#python agent.py --agent-class GoalFollower
#python ppo_agents.py --model-path "/models/checkpoints_mp3d/ckpt.849.pth" --input-type rgbd
# RGBD
python ppo_agents.py --model-path "/models/checkpoints_gibson/ckpt.1178.pth" --input-type rgbd