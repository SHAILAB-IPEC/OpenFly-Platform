import socket
import time
import threading
import argparse
from datetime import datetime

import numpy as np  
import yaml

from ue_bridge import UEBridge
from airsim_bridge import AirsimBridge
from gs_bridge import GSBridge


class TCPServer:
    def __init__(self, host='0.0.0.0', port=9999):
        self.server_address = (host, port)
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind(self.server_address)
        self.server_socket.listen(1)
        print(f"Listening on {host}:{port}")
    
    def accept_connection(self):
        conn, addr = self.server_socket.accept()
        print(f"Connected by {addr}")
        return conn
    
    def receive_pose(self, conn):

        data = conn.recv(1024).decode()
        return data


def load_config(config_file="config.yaml"):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config['thread_params']



def handle_planner(config_params, env_name):

    if 'ue' in env_name:
        env_bridge = UEBridge(config_params['sim_ip'], config_params['sim_port'], env_name)
    elif 'airsim' in env_name:
        env_bridge = AirsimBridge(env_name)
    elif 'gs' in env_name:
        env_bridge = GSBridge(env_name)
    else:
        print("Invalid env name!")
        return

    tcp_server = TCPServer(port=config_params['aim_port'])

    file_path = "uav_vln_data/test/"

    image_id = 0

    while True:

        print("ready to be connected")
        conn = tcp_server.accept_connection()
        print("Connection established!")

        while True:

            data = tcp_server.receive_pose(conn)
            if not data:
                break 

            if "path:" in data:
                file_path = data.split("path:")[1]
                image_id = 0
                print("file_path:", file_path)
                continue

            pose = list(map(float, data.split(',')))
            if len(pose) != 6:
                print("Invalid pose data!")
                continue
            print(f"Received pose: {pose}")
            
            if 'gs' in env_name:
                env_bridge.set_camera_pose(*pose, file_path)
            else:
                env_bridge.set_camera_pose(*pose)

            time.sleep(0.1)

            current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S_")

            env_bridge.process_camera_data(file_path = file_path + current_time_str +  str(image_id) + '.png')
            # env_bridge.process_camera_data(file_path = file_path +'0.png')
            image_id += 1
            print("Image saved!")


def main():
    
    parser = argparse.ArgumentParser(description="env name")
    parser.add_argument('--env', type=str, default='env_gs_sjtu02', help="input env name")

    args = parser.parse_args()
    config_file = "configs/" + args.env + ".yaml"

    planner_configs = load_config(config_file)

    threads = []
    for config in planner_configs:
        thread = threading.Thread(target=handle_planner, args=(config, args.env))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

if __name__ == "__main__":
    main()

