#!/bin/bash

RED='\e[31m'
GREEN='\e[32m'
YELLOW='\e[33m'
BLUE='\e[34m'
RESET='\e[0m' 

conda deactivate

# source setup 
source tool_ws/install/setup.sh
# set env name


# launch ros file
if [ -z "$1" ]; then
  echo -e "${RED}Error: You must provide an environment name.${RESET}"
  echo "Usage: $0 env_name bev|manual"
  exit 1
fi

export ENV=$1

# launch ros file
ros2 launch tool_ws/src/traj_gen/launch/traj_gen_launch.py