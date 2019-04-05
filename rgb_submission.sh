#!/usr/bin/env bash

#python agent.py --agent-class GoalFollower
#python ppo_agent.py --model-path "../data/checkpoints_mp3d/ckpt.849.pth" --input-type rgbd
#python ppo_agent.py --model-path "../data/checkpoints_gibson/ckpt.1178.pth" --input-type rgbd
# RGB
python ppo_agents.py --model-path "/models/checkpoints_gibson/ckpt.1178.pth" --input-type rgb