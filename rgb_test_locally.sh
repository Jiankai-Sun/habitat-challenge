#!/usr/bin/env bash

DOCKER_NAME="rgb_submission:v0.01"

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

nvidia-docker run -v /data/sunjiankai/Dataset/habitat_data/habitat-challenge-data/:/habitat-challenge-data \
    ${DOCKER_NAME} \
    /bin/bash -c \
    ". activate habitat; export CHALLENGE_CONFIG_FILE=/habitat-challenge-data/challenge_pointnav.local.rgb.yaml; bash submission.sh"

