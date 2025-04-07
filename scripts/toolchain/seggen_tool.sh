#!/bin/bash

RED='\e[31m'
GREEN='\e[32m'
YELLOW='\e[33m'
BLUE='\e[34m'
RESET='\e[0m' 


# conda deactivate

source tool_ws/install/setup.bash

# launch ros file
if [ -z "$1" ]; then
  echo -e "${RED}Error: You must provide an environment name. ${RESET}"
  echo "Usage: $0 env_name bev|manual"
  exit 1
fi

export ENV=$1 

if [[ "$ENV" == *"gs"* ]]; then
  echo -e "${YELLOW}Detected GS in environment name, Use manual annotation method ${RESET}"
  ros2 launch tool_ws/src/seg_gen/launch/manual_seg_launch.py
  exit 1
fi

if [[ "$ENV" == *"ue"* ]]; then
  echo -e "${YELLOW}In the UE env, we have collected precise landmarks based on Unreal Engine 5, and it is not recommended to perform further generation. ${RESET}"
  exit 1
fi

if [ -z "$2" ]; then
  echo -e "${YELLOW}Use the default BEV generation method ${RESET}"
  ros2 launch tool_ws/src/seg_gen/launch/bev_seg_launch.py
  sleep 0.5
  python3 tool_ws/src/seg_gen/scripts/bev_seg_gen.py --env "$ENV"
fi

seg_mode=$2

case "$seg_mode" in
  "bev")
    ros2 launch tool_ws/src/seg_gen/launch/bev_seg_launch.py
    sleep 0.5
    python3 tool_ws/src/seg_gen/scripts/bev_seg_gen.py --env "$ENV"
    ;;
  "manual")
    ros2 launch tool_ws/src/seg_gen/launch/manual_seg_launch.py
    ;;
  *)
    echo "Usage: $0 <env_name> <launch_file>"
    ;;
esac