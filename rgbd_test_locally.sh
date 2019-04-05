#!/usr/bin/env bash

DOCKER_NAME="rgbd_submission:v0.01"

while [[ $# -gt 0 ]]
do
key="${1}"

case $key in
      --docker-name)
      shift
      DOCKER_NAME="${1}"
	  shift
      ;;
    *) # unknown arg
      echo unkown arg ${1}
      exit
      ;;
esac
done

# Train on Gibson, test on Gibson
nvidia-docker run -v /data/sunjiankai/Dataset/habitat_data/habitat-challenge-data/:/habitat-challenge-data \
    ${DOCKER_NAME} \
    /bin/bash -c \
    ". activate habitat; export CHALLENGE_CONFIG_FILE=/habitat-challenge-data/challenge_pointnav.local.rgbd.yaml; bash submission.sh"


## Train on Gibson, test on Gibson
#nvidia-docker run -v /data/sunjiankai/Dataset/habitat_data/habitat-challenge-data/:/habitat-challenge-data \
#    ${DOCKER_NAME} \
#    /bin/bash -c \
#    ". activate habitat; export CHALLENGE_CONFIG_FILE=/habitat-challenge-data/challenge_pointnav.local.rgbd.yaml; bash python agent.py --model-path \"../data/checkpoints_gibson/ckpt.1178.pth\" --input-type rgbd"

## Train on mp3d, test on gibson
#nvidia-docker run -v /data/sunjiankai/Dataset/habitat_data/habitat-challenge-data/:/habitat-challenge-data \
#    ${DOCKER_NAME} \
#    /bin/bash -c \
#    ". activate habitat; export CHALLENGE_CONFIG_FILE=/habitat-challenge-data/challenge_pointnav.local.rgbd.yaml; bash python agent.py --model-path \"../data/checkpoints_mp3d/ckpt.849.pth\" --input-type rgbd"
