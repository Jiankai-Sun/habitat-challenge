#!/usr/bin/env bash
# pansy
#evalai set_token 698b115dc66280f246ef8d21ddc623ca3c5d6b49
# bowen
#evalai set_token e4885826fb8c486a6d6cefed212fe5064d3700ce
# cnmooc001
evalai set_token 9bf534b1a9f537145c912210fddb19a7693afbfc

# Push docker image to EvalAI docker registry
# RGBD
#evalai push rgbd_submission:v0.02 --phase habitat19-rgbd-minival
#evalai push rgbd_submission:v0.02 --phase habitat19-rgbd-test-std
evalai push rgbd_submission:v0.02 --phase habitat19-rgbd-test-ch
# RGB
# evalai push rgb_submission:v0.02 --phase habitat19-rgb-val