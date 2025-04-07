#!/bin/bash

RED='\e[31m'
GREEN='\e[32m'
YELLOW='\e[33m'
BLUE='\e[34m'
RESET='\e[0m' 



# launch ros file
if [ -z "$1" ]; then
  echo -e "${RED}Error: You must provide an environment name. ${RESET}"
  echo "Usage: $0 env_name"
  exit 1
fi

export ENV=$1 

if [[ "$ENV" == *"gs"* ]]; then
  echo -e "${YELLOW} The GS scene already has a point cloud data when it is generated ${RESET}"
  exit 1
fi

if [[ "$ENV" == *"ue"* ]]; then
  echo -e "${YELLOW}The UE scene already has a point cloud data when it is generated ${RESET}"
  exit 1
fi


python3 tool_ws/src/pcd_gen/scripts/airsim_pointcloud.py --env "$ENV"

