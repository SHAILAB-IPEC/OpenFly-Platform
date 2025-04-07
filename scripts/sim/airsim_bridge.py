import argparse
import io
import math
import os
import socket
import subprocess
import threading
import time
from datetime import datetime

import airsim
import cv2  # OpenCV
import numpy as np
import yaml
from common import *


class AirsimBridge:
    def __init__(self, env_name):
        self.env_name = env_name
        self._sim_thread = threading.Thread(target=self._init_airsim_sim)
        self._sim_thread.start()
        time.sleep(10)

        self._client = airsim.MultirotorClient()
        self._client.confirmConnection()
        self._client.enableApiControl(True)
        self._client.armDisarm(True)

    def _init_airsim_sim(self):
        env_dir = "envs/airsim/" + self.env_name

        if not os.path.exists(env_dir):
            raise ValueError(f"Specified directory {env_dir} does not exist")
        
        command = ["bash", f"{env_dir}/LinuxNoEditor/start.sh"]
        self.process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = self.process.communicate()
        print("Command output:\n", stdout)

    def set_camera_pose(self, x, y, z, pitch, yaw, roll):
        target_pose = airsim.Pose(airsim.Vector3r(x, -y, -z),
                                  airsim.to_quaternion(0, 0, math.radians(-yaw)))
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        self._client.simSetVehiclePose(target_pose, True)

    def set_drone_pos(self, x, y, z, pitch, yaw, roll):
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        qua = euler_to_quaternion(pitch, -yaw, roll)
        target_pose = airsim.Pose(airsim.Vector3r(x, y, z),
                                  airsim.Quaternionr(qua[0], qua[1], qua[2], qua[3]))
        self._client.simSetVehiclePose(target_pose, True)
        self._client.moveByVelocityBodyFrameAsync(0, 0, 0, 0.02)
        time.sleep(0.1)

    def _camera_init(self):
        '''Camera initialization'''
        camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(math.radians(15), 0, 0))
        self._client.simSetCameraPose("0", camera_pose)
        time.sleep(1)

    def _drone_init(self):
        '''Drone initialization'''
        self.set_drone_pos(0, 0, 0, 0, 0, 0)
        time.sleep(1)

    def get_camera_data(self, camera_type):
        valid_types = {'color', 'object_mask', 'depth'}
        if camera_type not in valid_types:
            raise ValueError(f"Invalid camera type. Expected one of {valid_types}, but got '{camera_type}'.")

        if camera_type == 'color':
            image_type = airsim.ImageType.Scene
        elif camera_type == 'depth':
            image_type = airsim.ImageType.DepthPlanar
        else:
            image_type = airsim.ImageType.Segmentation

        responses = self._client.simGetImages([airsim.ImageRequest('front_custom', image_type, False, False)])
        response = responses[0]
        if response.pixels_as_float:
            img_data = np.array(response.image_data_float, dtype=np.float32)
            img_data = np.reshape(img_data, (response.height, response.width))
        else:
            img_data = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_data = img_data.reshape(response.height, response.width, 3)

        return img_data

    def save_image(self, image_data, file_path):
        cv2.imwrite(file_path, image_data)

    def process_camera_data(self, file_path, camera_type='color'):
        img = self.get_camera_data(camera_type)
        self.save_image(img, file_path)
        print("Image saved")


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
        print("Data received:", data)
        return data


def handle_planner(config_params, env_name):
    airsim_bridge = AirsimBridge(env_name)
    tcp_server = TCPServer(port=config_params['aim_port'])

    file_path = "uav_vln_data/test/"
    image_id = 0
    while True:
        print("Ready to connect")
        conn = tcp_server.accept_connection()
        print("Connection established!")

        while True:
            data = tcp_server.receive_pose(conn)
            if not data:
                break

            if "path:" in data:
                file_path = data.split("path:")[1]
                image_id = 0
                print("File path:", file_path)
                continue

            pose = list(map(float, data.split(',')))
            print(f"Received pose: {pose}")

            airsim_bridge.set_camera_pose(*pose)
            time.sleep(0.1)
            current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S_")
            airsim_bridge.process_camera_data(file_path=file_path + current_time_str + str(image_id) + '.png')
            image_id += 1
            print("Image saved!")


def load_config(config_file="config.yaml"):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config['thread_params']


def main():
    parser = argparse.ArgumentParser(description="Environment name")
    parser.add_argument('--env', type=str, default='env_airsim_18', help="input env name")
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


if __name__ == '__main__':
    main()