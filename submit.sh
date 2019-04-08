#!/usr/bin/env bash
evalai set_token b15a5fa57eab65823131b788c76d4cf1bad726a7

# Push docker image to EvalAI docker registry
# RGBD
evalai push rgbd_submission:v0.01 --phase habitat19-rgbd-val
# RGB
# evalai push rgb_submission:v0.01 --phase habitat19-rgb-val