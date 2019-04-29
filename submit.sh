#!/usr/bin/env bash
evalai set_token 698b115dc66280f246ef8d21ddc623ca3c5d6b49

# Push docker image to EvalAI docker registry
# RGBD
evalai push rgbd_submission:v0.02 --phase habitat19-rgbd-minival
# RGB
# evalai push rgb_submission:v0.02 --phase habitat19-rgb-val