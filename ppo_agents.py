#!/usr/bin/env python3

# Actually this is SLAM Agent

import sys
sys.path.insert(0, '/ppo_baseline/map_and_plan_agent/')
import argparse

import habitat
from habitat import Config
from ppo_baseline.map_and_plan_agent.slam import DepthMapperAndPlanner


def get_defaut_config():
    c = Config()
    c.INPUT_TYPE = "blind"
    c.MODEL_PATH = "data/checkpoints/blind.pth"
    c.RESOLUTION = 256
    c.HIDDEN_SIZE = 512
    c.RANDOM_SEED = 7
    c.PTH_GPU_ID = 0
    return c

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-type",
        default="blind",
        choices=["blind", "rgb", "depth", "rgbd"],
    )
    parser.add_argument("--model-path", default="", type=str)
    args = parser.parse_args()

    config = get_defaut_config()
    config.INPUT_TYPE = args.input_type
    config.MODEL_PATH = args.model_path

    agent = DepthMapperAndPlanner(map_size_cm=1200, out_dir=None, mark_locs=True,
                                  reset_if_drift=True, count=-1, close_small_openings=True,
                                  recover_on_collision=True, fix_thrashing=True, goal_f=1.1, point_cnt=2)

    challenge = habitat.Challenge()
    challenge.submit(agent)


if __name__ == "__main__":
    main()
